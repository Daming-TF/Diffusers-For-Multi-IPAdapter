import argparse
import random
import json
import os
from insightface.app import FaceAnalysis
import multiprocessing
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file_dir = '/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/_tmp/log'
os.makedirs(log_file_dir, exist_ok=True)
log_file_path = f'{log_file_dir}/202402090333.log'
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


data_dict = {
    'Laion': ['data-50m', 'data-50m-V1_face_all_info'],
    'coyo': ['coyo700m/data', 'coyo700m/data_V1_face_all_info'],
    'ffhq': ['in-the-wild-images', 'in-the-wild-images_V1_face_all_info'],
}


def json_transfer():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    error_num = 0
    error_image_record_dir = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/_tmp/error_image/"
    error_image_record_paths = [os.path.join(error_image_record_dir, name) for name in os.listdir(error_image_record_dir)]
    for i, json_path in enumerate(error_image_record_paths):
        with open(json_path, 'r')as f:
            data = json.load(f)
            error_num += len(data)

    with open(args.json_path, 'r')as f:
        meta_data_list = json.load(f)

    result = []
    no_exists_num = 0
    for meta_data in tqdm(meta_data_list):
        image_path = meta_data["image_file"]
        find_status = False
        for data in data_dict.keys():
            if data in image_path:
                find_status = True
                break
        assert find_status, ValueError("some error is happened")
        suffix = os.path.basename(image_path).split('.')[-1]
        json_path = image_path.replace(data_dict[data][0], data_dict[data][1]).replace(suffix, 'json')
        # print(f"image path:{image_path}\njson path :{json_path}")
        # assert os.path.exists(json_path), ValueError(f"json file is not exists ==> {json_path}")
        if not os.path.exists(json_path):
            no_exists_num += 1
            continue

        meta_data["face_info_json"] = json_path
        result.append(meta_data)

    assert error_num==no_exists_num, ValueError(f"some error has happened\t{error_num}\t{no_exists_num}")
    with open(args.save_path, "w")as f:
        json.dump(result, f)
    print(f"result has saved in {args.save_path}")


def processing(i, data):
    app = FaceAnalysis(name='/home/mingjiahui/.insightface/models/antelopev2/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    transfer = transforms.Resize(512)
    error_image = []
    error_save_path = f"/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/_tmp/error_image/{i}.json"

    for data_ in tqdm(data):
        meta_result = {}
        meta_result.setdefault('landmark_2d_106', [])
        meta_result.setdefault('landmark_3d_68', [])
        meta_result.setdefault('kps', [])
        meta_result.setdefault('pose', [])
        meta_result.setdefault('sex', [])
        meta_result.setdefault('age', [])
        meta_result.setdefault('bbox', [])
        image_path = data_['image_file']
        embeds_paths = data_['embeds_path']
        assert os.path.exists(embeds_paths)
        
        find_status = False
        for dt in data_dict.keys():
            if dt in image_path:
                find_status = True
                break
        assert find_status is True, ValueError(f"some error happend ==> {image_path}")
        suffix = os.path.basename(image_path).split('.')[-1]
        save_path = image_path.replace(data_dict[dt][0], data_dict[dt][1]).replace(suffix, 'json')
        if os.path.exists(save_path):
            continue
        
        landmark_input = Image.open(image_path).convert("RGB")
        ori_w, ori_h = landmark_input.size
        landmark_input = transfer(landmark_input)
        new_w, new_h = landmark_input.size
        ratio = ori_w / new_w
        face_info = app.get(cv2.cvtColor(np.array(landmark_input), cv2.COLOR_RGB2BGR))

        if len(face_info)==0:
            logger.info(f"This img has not face info ==> {image_path}")
            error_image.append(image_path)
            continue
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]

        meta_result['image_path'] = image_path
        meta_result['landmark_2d_106'].append((face_info["landmark_2d_106"]*ratio).tolist())
        meta_result['landmark_3d_68'].append((face_info["landmark_2d_106"]*ratio).tolist())
        meta_result['kps'].append((face_info["kps"]*ratio).tolist())
        meta_result['pose'].append((face_info["pose"]*ratio).tolist())
        meta_result['sex'].append(face_info.sex)
        meta_result['age'].append(face_info["age"])
        meta_result['bbox'].append((face_info["bbox"]*ratio).tolist())

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w')as f:
            json.dump(meta_result, f)  
        # print(f"result has saved in {save_path}")  
        # exit(0)  

    os.makedirs(os.path.dirname(error_save_path), exist_ok=True)
    with open(error_save_path, 'w')as f:
        json.dump(error_image, f) 
    print(f"error image has saved in {error_save_path}")
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json//traindata_V1.json")
    parser.add_argument("--process_num", type=int, default=1)
    # parser.add_argument("output", type=str, default=None)
    args = parser.parse_args()
    with open(args.input_json, 'r')as f:
        data = json.load(f)
    print(f"Total num:{len(data)}")
    random.shuffle(data)
    chunk_num = len(data) // args.process_num
    residue_num = len(data) % args.process_num

    process_list = []
    data_index = 0
    for i in range(args.process_num):
        if i < residue_num:
            chunk_input = data[data_index: data_index+chunk_num+1]
            data_index += chunk_num+1
        else:
            chunk_input = data[data_index: data_index+chunk_num]
            data_index += chunk_num
        
        process = multiprocessing.Process(target=processing, args=(i, chunk_input))
        process.start()
        process_list.append(process)
    
    for process in process_list:
        process.join()
        