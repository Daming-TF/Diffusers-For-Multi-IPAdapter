from fastparquet import ParquetFile
import sys
import os
from tqdm import tqdm
import requests
import json
import argparse
import multiprocessing
from requests.exceptions import Timeout
import hashlib
import torch
from PIL import Image
import cv2
import numpy as np
import random
from datetime import datetime

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import psutil

source_dir = "/mnt/nfs/file_server/public/xy/data/unsplash_lite"


def face_dect(i, img_paths, tmp_dir=None):
    save_dir_key = [
        "unsplash_lite/imgs",
        "unsplash_lite/imgs_embeds",
    ]

    tmp0_dir = f"{source_dir}/_tmp/no_face_detected"
    tmp0_path = os.path.join(tmp0_dir, f'{i}.json')
    no_face_record = []

    tmp1_dir = f"{source_dir}/_tmp/error_img"
    tmp1_path = os.path.join(tmp1_dir, f'{i}.json')
    error_img = []
    if not os.path.exists(tmp1_dir):
        # record no face data
        os.makedirs(tmp0_dir, exist_ok=True)
        with open(tmp0_path, 'w')as f:
            json.dump(no_face_record, f) 
    

    current_datetime = datetime.now()
    current_year = current_datetime.year
    current_month = current_datetime.month
    current_day = current_datetime.day
    tmp2_dir = f"{source_dir}/_tmp/multi_face"
    tmp3_dir = f"{source_dir}/_tmp/wface"
    tmp2_path = os.path.join(tmp2_dir, f'{i}.json')
    tmp3_path = os.path.join(tmp3_dir, f'{i}.json')
    w_face = []
    multi_face = []

    total = 0
    exist = 0
    success = 0
    freq = 1000

    from insightface.app import FaceAnalysis
    from torchvision import transforms
    app = FaceAnalysis(name='/home/mingjiahui/.insightface/models/antelopev2/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    transfer = transforms.Resize(768)

    for img_path in tqdm(img_paths):
        total += 1
        meta_result = {}
        meta_result.setdefault('landmark_2d_106', [])
        meta_result.setdefault('landmark_3d_68', [])
        meta_result.setdefault('kps', [])
        meta_result.setdefault('pose', [])
        meta_result.setdefault('sex', [])
        meta_result.setdefault('age', [])
        meta_result.setdefault('bbox', [])
        meta_result.setdefault('embeds', [])
        
        suffix = os.path.basename(img_path).split('.')[1]
        json_save_path = img_path.replace(save_dir_key[0], save_dir_key[1]).replace(suffix, 'json')

        #### debug
        # print(f"tmp0_path:{tmp0_path}\n\
        #         tmp1_path:{tmp1_path}\n\
        #         tmp2_path:{tmp2_path}\n\
        #         tmp3_path:{tmp3_path}\n\
        #         json_save_path:{json_save_path}\n\
        #         img_path:{img_path}")
        # exit(0)

        os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
        if os.path.exists(json_save_path):
            exist += 1
            if total % freq == 0:
                print(f"Total:{total}\tSuccess:{success}\tError:{len(error_img)}\tnoface:{len(no_face_record)}\tmulti:{len(multi_face)}\tExist{exist}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # print(e)
            error_img.append(img_path)
            if total % freq == 0:
                print(f"Total:{total}\tSuccess:{success}\tError:{len(error_img)}\tnoface:{len(no_face_record)}\tmulti:{len(multi_face)}\tExist{exist}")
            continue
        ori_w, _ = img.size
        img = transfer(img)
        new_w, _ = img.size
        ratio = ori_w / new_w
        face_info = app.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        if len(face_info)==0:
            logger.info(f"This img has not face info ==> {img_path}")
            no_face_record.append(img_path)
            if total % freq == 0:
                print(f"Total:{total}\tSuccess:{success}\tError:{len(error_img)}\tnoface:{len(no_face_record)}\tmulti:{len(multi_face)}\tExist{exist}")
            continue
        elif len(face_info) > 1:
            multi_face.append(img_path)
        else:
            w_face.append(img_path)

        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        
        meta_result['image_file'] = img_path
        meta_result['landmark_2d_106'].append((face_info["landmark_2d_106"]*ratio).tolist())
        meta_result['landmark_3d_68'].append((face_info["landmark_3d_68"]*ratio).tolist())
        meta_result['kps'].append((face_info["kps"]*ratio).tolist())
        meta_result['pose'].append((face_info["pose"]*ratio).tolist())
        meta_result['sex'].append(face_info.sex)
        meta_result['age'].append(face_info["age"])
        meta_result['bbox'].append((face_info["bbox"]*ratio).tolist())

        face_emb = torch.from_numpy(face_info.embedding).unsqueeze(0)
        cache_save_dir = os.path.dirname(json_save_path)
        cache_save_name = os.path.basename(json_save_path).replace('json', 'npy')
        cache_path = os.path.join(cache_save_dir, cache_save_name)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, face_emb)
        meta_result['embeds'].append(cache_path)

        os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
        with open(json_save_path, 'w')as f:
            json.dump(meta_result, f)  
        success += 1

        if total % freq == 0:
            print(f"Total:{total}\tSuccess:{success}\tError:{len(error_img)}\tnoface:{len(no_face_record)}\tmulti:{len(multi_face)}\tExist{exist}")

    # record no face data
    os.makedirs(tmp0_dir, exist_ok=True)
    with open(tmp0_path, 'w')as f:
        json.dump(no_face_record, f)  
    
    # record error data
    os.makedirs(tmp1_dir, exist_ok=True)
    with open(tmp1_path, 'w')as f:
        json.dump(error_img, f)  
    
    # record multi data
    os.makedirs(tmp2_dir, exist_ok=True)
    with open(tmp2_path, 'w')as f:
        json.dump(multi_face, f)  

    # record wface data
    os.makedirs(tmp3_dir, exist_ok=True)
    with open(tmp3_path, 'w')as f:
        json.dump(w_face, f)  
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_num", type=int, default=1)
    parser.add_argument("--min_reso", type=int, default=512)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--max_wait_time", type=int, default=5)
    args = parser.parse_args()

    tmp_dir = os.path.join(source_dir, '_tmp', 'wface')
    save_path = os.path.join(tmp_dir, 'total.json')
    img_dir = os.path.join(source_dir, 'imgs')
    img_paths = [os.path.join(img_dir, name) for name in os.listdir(img_dir) if name.endswith('.png')]
    print(f"Total data num :{len(img_paths)}")

    processors = []
    chunk_num = len(img_paths) // args.process_num
    residue_num = len(img_paths) % args.process_num
    data_index = 0
    for i in range(args.process_num):
        if i < residue_num:
            chunk_data = img_paths[data_index:data_index+chunk_num+1]
            data_index = data_index+chunk_num+1
        else:
            chunk_data = img_paths[data_index:data_index+chunk_num]
            data_index = data_index+chunk_num
        processor = multiprocessing.Process(target=face_dect, args=(i, chunk_data))
        processors.append(processor)
        processor.start()
    for processor in processors:
        processor.join()

    result = []
    tmp_file_paths = [os.path.join(tmp_dir, name) for name in os.listdir(tmp_dir)]
    for tmp_file_path in tmp_file_paths:
        with open(tmp_file_path, 'r')as f:
            result += json.load(f)
    with open(save_path, 'w')as f:
        json.dump(result, f)
    print(f"result has saved in {save_path}")
