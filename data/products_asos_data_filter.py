import pandas as pd
import os
import json
from tqdm import tqdm
import requests
import argparse
import multiprocessing
import math

import cv2
from PIL import Image
import numpy as np
import torch

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def face_dect(img_paths):
    save_dir_key = [
        "/TrainingDataPro--asos-e-commerce-dataset/data",
        "/TrainingDataPro--asos-e-commerce-dataset/data_antelopev2_embeds",
    ]
    from insightface.app import FaceAnalysis
    from torchvision import transforms
    app = FaceAnalysis(name='/home/mingjiahui/.insightface/models/antelopev2/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    transfer = transforms.Resize(1024)

    for img_path in tqdm(img_paths):
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
        os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
        
        if os.path.exists(json_save_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(e)
            continue
        ori_w, ori_h = img.size
        img = transfer(img)
        new_w, new_h = img.size
        ratio = ori_w / new_w
        face_info = app.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        if len(face_info)==0:
            logger.info(f"This img has not face info ==> {img_path}")
            continue
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]

        meta_result['image_file'] = img_path
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
        cache_save_name = os.path.basename(json_save_path).replace('json', 'npy')
        cache_path = os.path.join(cache_save_dir, cache_save_name)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, face_emb)
        meta_result['embeds'].append(cache_path)

        os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
        with open(json_save_path, 'w')as f:
            json.dump(meta_result, f)  
        # print(f"result has saved in {json_save_path}")
        # break


def load_csv(csv_file):
    df = pd.read_csv(csv_file, encoding='gbk')
    urls = df['images'].values
    # name = df['name'].values
    description = df['description'].values
    color = df['color'].values
    category = df['category'].values
    sku = df['sku'].values
    return urls, description, color, category, sku


def download_file(url, save_path, max_attempts=10):
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url)
            with open(save_path, 'wb') as file:
                file.write(response.content)
            return 0
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt} failed:", e)
    print("Max attempts reached. Failed to download the URL:", url)
    return 1
    


def download_process(i, urls, description, color, category, sku, args):
    save_dir = os.path.join(args.data_dir, str(i).zfill(6))
    max_num = args.max_attempts
    for urls_, description_, color_, category_, sku_ in tqdm(zip(urls, description, color, category, sku), total=len(urls)):
        prompt = f"A person in a {color_} {category_}"
        if math.isnan(sku_):
            continue
        i = 10
        for i_ in range(i):
            save_dir_ = os.path.join(save_dir, f'{str(int(sku_))}_{i_}')
            if not os.path.exists(save_dir_):
                break
        os.makedirs(save_dir_, exist_ok=True)
        
        urls_ = urls_.replace('[', '').replace(']', '').replace('\'', '').split(',')
        for j, url_ in enumerate(urls_[:2]):
            img_path = os.path.join(save_dir_, f"{str(j).zfill(6)}.jpg")
            if download_file(url_, img_path, max_attempts=max_num):
                continue
            result = {
                'image_path': img_path,
                'prompt': prompt,
                'description': description_,
            }
            txt_path = os.path.join(save_dir_, f"{str(j).zfill(6)}.txt")
            json_path = os.path.join(save_dir_, f"{str(j).zfill(6)}.json")
            with open(txt_path, 'w')as f:
                f.write(prompt)
            with open(json_path, 'w')as f:
                json.dump(result, f)
            # print(f"result has saved in {json_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="/mnt/nfs/file_server/public/mingjiahui/data/TrainingDataPro--asos-e-commerce-dataset/products_asos.csv")
    parser.add_argument("--data_dir", type=str, default="/mnt/nfs/file_server/public/mingjiahui/data/TrainingDataPro--asos-e-commerce-dataset/data")
    parser.add_argument("--process_num", type=int, default=1)
    parser.add_argument("--max_attempts", type=int, default=10)
    args = parser.parse_args()


    # #### download
    # urls, description, color, category, sku = load_csv(args.csv_file)
    # assert len(urls)==len(description)==len(color)==len(category)==len(sku), ValueError("some error happened")

    # data_index = 0
    # processors = []
    # chunk_num = len(urls) // args.process_num
    # residue_num = len(urls) % args.process_num
    # for i in range(args.process_num):
    #     end_index = data_index+chunk_num+1 if i < residue_num else data_index+chunk_num

    #     chunk_urls = urls[data_index:end_index]
    #     chunk_description = description[data_index:end_index]
    #     chunk_color = color[data_index:end_index]
    #     chunk_category = category[data_index:end_index]
    #     chunk_sku = sku[data_index:end_index]

    #     processor = multiprocessing.Process(
    #         target=download_process, 
    #         args=(i, chunk_urls, chunk_description, chunk_color, chunk_category, chunk_sku, args))
    #     processors.append(processor)
    #     processor.start()
    # for processor in processors:
    #     processor.join()


    #### face detect
    source_path = "/mnt/nfs/file_server/public/mingjiahui/data/TrainingDataPro--asos-e-commerce-dataset/data"
    process_dirs = [os.path.join(source_path, name) for name in os.listdir(source_path)]
    img_paths = []
    for process_dir in tqdm(process_dirs):
        img_dirs = [os.path.join(process_dir, name) for name in os.listdir(process_dir)]
        for img_dir in img_dirs:
            img_paths += [os.path.join(img_dir, name) for name in os.listdir(img_dir) if name.endswith('.jpg')]
    print(f"total num:{len(img_paths)}")
    
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
        processor = multiprocessing.Process(target=face_dect, args=(chunk_data,))
        processors.append(processor)
        processor.start()
    for processor in processors:
        processor.join()
    # ++++++++++++++++++++++++++++++++++++++++++++++