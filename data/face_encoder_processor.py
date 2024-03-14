from insightface.app import FaceAnalysis
import json
from argparse import ArgumentParser
import random
import multiprocessing
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import torch
import os


def antelopev2_processor(i, chunk_input):
    data_dict = {
        'Laion': ['data-50m', 'data-50m_antelopev2'],
        'coyo': ['coyo700m/data', 'coyo700m/data_antelopev2'],
        'ffhq': ['in-the-wild-images', 'in-the-wild-images_antelopev2'],
    }
    transform = transforms.Resize(1024)
    error_save_path = f"/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/_tmp/antelopev2_error_image/{i}.json"
    os.makedirs(os.path.dirname(error_save_path), exist_ok=True)
    error_image= []

    app = FaceAnalysis(name='/home/mingjiahui/.insightface/models/antelopev2/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    for metadata in tqdm(chunk_input):
        # init
        image_path = metadata['image_file']
        find_status = False
        for data in data_dict.keys():
            if data in image_path:
                find_status = True
                break
        assert find_status, ValueError(f"some error happened ++> {image_path}")
        suffix = os.path.basename(image_path).split('.')[-1]
        save_path = image_path.replace(data_dict[data][0], data_dict[data][1]).replace(suffix, 'npy')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # if os.path.exists(save_path):
        #     continue

        # get face info
        face_image = transform(Image.open(image_path).convert("RGB"))
        face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        if len(face_info) == 0:
            error_image.append(image_path)
            print(f"this image has no face info ==> {image_path}")
            continue
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]   # only use the maximum face
        face_emb = torch.from_numpy(face_info.normed_embedding).unsqueeze(0)

        # save
        np.save(save_path, face_emb)
        # print(f"result has saved in ++> {save_path}")
        # exit(0)
    
    with open(error_save_path, 'w')as f:
        json.dump(error_image, f)
        print(f"error image info has saved in {error_save_path}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_json", type=str, default="/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1.json")
    parser.add_argument("--process_num", type=int, default=1)
    parser.add_argument("--image_encoder", type=str, default='antelopev2', help="Union['antelopev2', ]")
    args = parser.parse_args()

    assert args.image_encoder in ['antelopev2'], \
        ValueError(f"the param '--image encoder' <++> '{args.image_encoder}' is not support")
    if args.image_encoder == 'antelopev2':
        target = antelopev2_processor
    

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
        process = multiprocessing.Process(target=target, args=(i, chunk_input))
        process.start()
        process_list.append(process)
    
    for process in process_list:
        process.join()
    