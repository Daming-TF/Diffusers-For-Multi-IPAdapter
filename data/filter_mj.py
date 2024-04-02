import json
from tqdm import tqdm
import os
from PIL import Image
import cv2
import numpy as np
import logging
import torch
import multiprocessing
import random
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def statistic_reso(data, save_path):
    for data_ in tqdm(data):
        image_path = data_['image_file']
        img = Image.open(image_path)
        if min(img.size) > 768:
            result.append(data_)
    
    with open(save_path, 'w')as f:
        json.dump(result, f)
    print(f"result has saved in {save_path}")
    

def get_mj_train_json():
    result = []
    source_dir = "/mnt/nfs/file_server/public/xy/data/data/mj_zip_data"
    save_dir = "/mnt/nfs/file_server/public/mingjiahui/data/MJ_xy"
    json_path = "/mnt/nfs/file_server/public/mingjiahui/data/MJ_xy/_tmp/traindata_V2_mjh_ori.json"
    save_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V2_mj_xy.json"
    with open(json_path, 'r')as f:
        data = json.load(f)
    for data_ in tqdm(data):
        result_ = {}
        image_name = data_['source']
        prompt = data_['prompt']

        image_path = os.path.join(source_dir, image_name)
        suffix = os.path.basename(image_path).split('.')[1]
        json_save_path = os.path.join(save_dir, os.path.dirname(image_name), os.path.basename(image_name).replace(suffix, 'json'))
        
        if not os.path.exists(json_save_path):
            continue
        try:
            with open(json_save_path, 'r') as f:
                face_info = json.load(f)
        except Exception as e:
            print(e)
            print(json_save_path)
            exit(0)
        if len(face_info['embeds']) != 1:
            continue
        assert os.path.exists(face_info['embeds'][0]), ValueError(f"{face_info['embeds'][0]} is not exists")
        result_['image_file'] = image_path
        result_['text'] = prompt
        result_['embeds_path'] = face_info['embeds'][0]
        result_['face_info_json'] = json_save_path

        result.append(result_)

    with open(save_path, 'w')as f:
        json.dump(result, f)
    print(f"result has saved in {save_path}")
    print(f"Total num:{len(result)}")
  

def face_dect(data):
    source_dir = "/mnt/nfs/file_server/public/xy/data/data/mj_zip_data"
    save_dir = "/mnt/nfs/file_server/public/mingjiahui/data/MJ_xy/non_norm_embeds"
    count = 0
    os.makedirs(save_dir, exist_ok=True)
    from insightface.app import FaceAnalysis
    from torchvision import transforms
    app = FaceAnalysis(name='/home/mingjiahui/.insightface/models/antelopev2/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    transfer = transforms.Resize(512)

    for data_ in tqdm(data):
        meta_result = {}
        meta_result.setdefault('landmark_2d_106', [])
        meta_result.setdefault('landmark_3d_68', [])
        meta_result.setdefault('kps', [])
        meta_result.setdefault('pose', [])
        meta_result.setdefault('sex', [])
        meta_result.setdefault('age', [])
        meta_result.setdefault('bbox', [])
        meta_result.setdefault('embeds', [])
        image_name = data_['source']
        prompt = data_['prompt']
        
        image_path = os.path.join(source_dir, image_name)
        suffix = os.path.basename(image_path).split('.')[1]
        json_save_path = os.path.join(save_dir, os.path.dirname(image_name), os.path.basename(image_name).replace(suffix, 'json'))
        
        if os.path.exists(json_save_path):
            continue

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(e)
            continue
        ori_w, ori_h = img.size
        img = transfer(img)
        new_w, new_h = img.size
        ratio = ori_w / new_w
        face_info = app.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        if len(face_info)==0:
            logger.info(f"This img has not face info ==> {image_path}")
            continue
        face_infos = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[::-1]

        meta_result['image_file'] = image_path
        meta_result['text'] = prompt
        for i, face_info in enumerate(face_infos): 
            meta_result['landmark_2d_106'].append((face_info["landmark_2d_106"]*ratio).tolist())
            meta_result['landmark_3d_68'].append((face_info["landmark_3d_68"]*ratio).tolist())
            meta_result['kps'].append((face_info["kps"]*ratio).tolist())
            meta_result['pose'].append((face_info["pose"]*ratio).tolist())
            meta_result['sex'].append(face_info.sex)
            meta_result['age'].append(face_info["age"])
            meta_result['bbox'].append((face_info["bbox"]*ratio).tolist())

            # face_emb = torch.from_numpy(face_info.normed_embedding).unsqueeze(0)
            face_emb = torch.from_numpy(face_info.embedding).unsqueeze(0)
            cache_save_dir = os.path.dirname(json_save_path)
            cache_save_name = os.path.basename(json_save_path).split('.')[0]+f'_{i}.npy'
            cache_path = os.path.join(cache_save_dir, cache_save_name)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, face_emb)
            meta_result['embeds'].append(cache_path)

        os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
        with open(json_save_path, 'w')as f:
            json.dump(meta_result, f)  
        
        count += 1 if len(face_infos)==1 else 0
        if count % 1000 == 0:
            print(count)


if __name__ == '__main__':
    #### get mj data json
    # result = []
    # save_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V2_mjh_ori.json"
    # json_path = "/mnt/nfs/file_server/public/xy/data/laion_aesthetic_6/shape_embcache_mj_laion_canny_lineart.json"
    # with open(json_path, 'r')as f:
    #     data = json.load(f)
    # for data_ in tqdm(data):
    #     if data_['type'] != 'mj-general':
    #         continue
    #     result.append(data_)
    # with open(save_path, 'w')as f:
    #     json.dump(result, f)
    # print(f"result has saved in {save_path} ==> Total num:{len(result)}")
    # # ++++++++++++++++++++++++++++++++++++++++++++++

    #### face detect
    json_path = "/mnt/nfs/file_server/public/mingjiahui/data/MJ_xy/_tmp/traindata_V2_mjh_ori.json"
    process_num = 4
    processors = []
    with open(json_path, 'r')as f:
        data = json.load(f)
        random.shuffle(data)
    chunk_num = len(data) // process_num
    residue_num = len(data) % process_num
    data_index = 0
    for i in range(process_num):
        if i < residue_num:
            chunk_data = data[data_index:data_index+chunk_num+1]
            data_index = data_index+chunk_num+1
        else:
            chunk_data = data[data_index:data_index+chunk_num]
            data_index = data_index+chunk_num
        processor = multiprocessing.Process(target=face_dect, args=(chunk_data,))
        processors.append(processor)
        processor.start()
    for processor in processors:
        processor.join()
    # ++++++++++++++++++++++++++++++++++++++++++++++

    # #### get mj train json
    # get_mj_train_json()

    # #### statistic_reso
    # result = []
    # processors = []
    # process_num = 4
    # save_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V2_mj_xy--min_reso_768.json"
    # tmp_save_dir = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/_tmp/mj_minreso_768"
    # os.makedirs(tmp_save_dir, exist_ok=True)
    # json_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V2_mj_xy.json"
    # with open(json_path, 'r')as f:
    #     data = json.load(f)
    # chunk_num = len(data) // process_num
    # residue_num = len(data) % process_num
    # data_index = 0
    # for i in range(process_num):
    #     if i < residue_num:
    #         chunk_data = data[data_index:data_index+chunk_num+1]
    #         data_index = data_index+chunk_num+1
    #     else:
    #         chunk_data = data[data_index:data_index+chunk_num]
    #         data_index = data_index+chunk_num
    #     tmp_path = os.path.join(tmp_save_dir, f"{i}.json")
    #     processor = multiprocessing.Process(target=statistic_reso, args=(chunk_data, tmp_path))
    #     processors.append(processor)
    #     processor.start()
    # for processor in processors:
    #     processor.join()

    # for i in range(process_num):
    #     tmp_json = os.path.join(tmp_save_dir, f"{i}.json")
    #     with open(tmp_json, 'r')as f:
    #         result += json.load(f)

    # with open(save_path, 'w')as f:
    #     json.dump(result, f)
    # print(f"result has saved in {save_path}")
    # print(f"Total num:{len(result)}")
   