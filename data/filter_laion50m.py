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

source_dir = "/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face"


def memory_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 * 1024)
    return mem

def list_files(directory):
    file_list = []
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.jpg'):
                file_list.append(entry.name)
    return file_list

def load_paths_(i, img_dirs, tmp_dir, endswith='.jpg'):
    os.makedirs(os.path.dirname(tmp_dir), exist_ok=True)
    total = 0
    for img_dir in tqdm(img_dirs):
        img_paths = []
        tmp_path = os.path.join(tmp_dir, f"{os.path.basename(img_dir)}.json")
        for name in tqdm(os.listdir(img_dir)):
            img_paths.append(os.path.join(img_dir, name)) if name.endswith(endswith) else None
        # img_paths += [os.path.join(img_dir, name) for name in list_files(img_dir)]
        with open(tmp_path, 'w')as f:
            json.dump(img_paths, f)
        total += len(img_paths)
    print(f"Process {i}: {total}")


def load_paths(source_path, endswith='.jpg', process_num=1):
    tmp_dir = f"{source_dir}/data-50m-20240402-embeds/_tmp/load_img_paths"
    img_dirs = [os.path.join(source_path, name) for name in os.listdir(source_path)]
    random.shuffle(img_dirs)
    processors = []
    # for i, img_dir in enumerate(img_dirs):
    data_index = 0
    chunk_num = len(img_dirs) // process_num
    residue_num = len(img_dirs) % process_num
    for i in range(process_num):
        end_index = data_index+chunk_num+1 if i < residue_num else data_index + chunk_num
        chunk_dir = img_dirs[data_index:end_index]
        data_index = end_index
        processor = multiprocessing.Process(target=load_paths_, args=(i, chunk_dir, tmp_dir, endswith))
        processors.append(processor)
        processor.start()
    for processor in processors:
        processor.join()

    img_paths = []
    for img_dir in img_dirs:
        tmp_path = os.path.join(tmp_dir, f"{os.path.basename(img_dir)}.json")
        with open(tmp_path, 'r')as f:
            img_paths += json.load(f)

    save_path = os.path.join(tmp_dir, "total_img.json")
    with open(save_path, 'w')as f:
        json.dump(img_paths, f)
    print(f"img paths has saved in {save_path}")
    return img_paths



def face_dect(i, img_paths, if_model='antelopev2'):
    save_dir_key = [
        "data-50m-20240402",
        f"data-50m-20240402-embeds/{if_model}_embeds",
    ]

    tmp0_dir = f"{source_dir}/data-50m-20240402-embeds/_tmp/no_face_detected"
    tmp0_path = os.path.join(tmp0_dir, f'{i}.json')
    no_face_record = []

    tmp1_dir = f"{source_dir}/data-50m-20240402-embeds/_tmp/error_img"
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
    tmp2_dir = f"{source_dir}/data-50m-20240402-embeds/_tmp/{current_year}_{current_month:02d}_{current_day:02d}--multi_face"
    tmp2_path = os.path.join(tmp2_dir, f'{i}.json')
    multi_face = []

    total = 0
    exist = 0
    success = 0
    freq = 1000

    from insightface.app import FaceAnalysis
    from torchvision import transforms
    app = FaceAnalysis(name=f'/home/mingjiahui/.insightface/models/{if_model}/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
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
        os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
        if os.path.exists(json_save_path):
            exist += 1
            if total % freq == 0:
                print(f"Total:{total}\tSuccess:{success}\tError:{len(error_img)}\tnoface:{len(no_face_record)}\tmulti:{len(multi_face)}\tExist{exist}\t{(exist+success)/total}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # print(e)
            error_img.append(img_path)
            if total % freq == 0:
                print(f"Total:{total}\tSuccess:{success}\tError:{len(error_img)}\tnoface:{len(no_face_record)}\tmulti:{len(multi_face)}\tExist{exist}\t{(exist+success)/total}")
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
                print(f"Total:{total}\tSuccess:{success}\tError:{len(error_img)}\tnoface:{len(no_face_record)}\tmulti:{len(multi_face)}\tExist{exist}\t{(exist+success)/total}")
            continue
        if len(face_info) > 1:
            multi_face.append(img_path)

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
            print(f"Total:{total}\tSuccess:{success}\tError:{len(error_img)}\tnoface:{len(no_face_record)}\tmulti:{len(multi_face)}\tExist{exist}\t{(exist+success)/total}")

    # record no face data
    os.makedirs(tmp0_dir, exist_ok=True)
    with open(tmp0_path, 'w')as f:
        json.dump(no_face_record, f)  
    
    # record no face data
    os.makedirs(tmp1_dir, exist_ok=True)
    with open(tmp1_path, 'w')as f:
        json.dump(error_img, f)  
    
    # record no face data
    os.makedirs(tmp2_dir, exist_ok=True)
    with open(tmp2_path, 'w')as f:
        json.dump(multi_face, f)  
            

def download_file(url, save_path, max_attempts=10, max_wait_time=5):
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, timeout=max_wait_time)
            with open(save_path, 'wb') as file:
                file.write(response.content)
            return 0
        except Timeout:
            print(f"timeout, {url}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt} failed:", e)
        
    print("Max attempts reached. Failed to download the URL:", url)
    return 1


def download_process(i, parquet_list, args):
    # urls, description, color, category, sku, args
    save_key=[
        "laion_face_meta",
        "data-50m-20240402"
    ]
    hash_object = hashlib.sha256()
    freq = 1000
    for parquet_file in parquet_list:
        success_count = 0
        fail_count = 0
        skip_count = 0
        exists = 0
        total = 0
        parquet_name = os.path.basename(parquet_file).split('.')[0]
        error_save_dir = os.path.join("/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/_tmp/failed_download_record", parquet_name)
        pf = ParquetFile(parquet_file)
        dF = pf.to_pandas()
        
        for index, row in tqdm(dF.iterrows(), total=len(dF)):
            total += 1
            sample_id = row['SAMPLE_ID']
            url = row['URL']
            caption = row['TEXT']
            ori_width = row['WIDTH']
            ori_height = row['HEIGHT']
            license = row['LICENSE']
            nsfw = row['NSFW']
            
            hash_object.update(
                str((sample_id, url, caption, ori_width, ori_height, license, nsfw)).encode()
                )
            hashed_pair = hash_object.hexdigest()
            
            try:
                min_reso = min(ori_width, ori_height)
                assert min_reso > args.min_reso, ValueError()
            except Exception as e:
                skip_count += 1
                if total % freq == 0:
                    print(f"Total:{total}\tSuccess:{success_count}\tFail:{fail_count}\tSkip:{skip_count}\texists:{exists}\t{round((success_count+exists)/total, 2)}")
                continue

            save_dir = parquet_file.replace(save_key[0], save_key[1]).replace('.parquet', '')
            os.makedirs(save_dir, exist_ok=True)
            # save_name = str(index).zfill(8)
            img_path  = os.path.join(save_dir, hashed_pair+'.jpg')
            txt_path  = os.path.join(save_dir, hashed_pair+'.txt')
            json_path  = os.path.join(save_dir, hashed_pair+'.json')
            error_save_path = os.path.join(error_save_dir, hashed_pair+'.json')
            if (os.path.exists(img_path) and os.path.exists(txt_path) and os.path.exists(json_path)) or os.path.exists(error_save_path):
                if os.path.exists(error_save_path):
                    fail_count += 1  
                else: 
                    exists += 1
                if total % freq == 0:
                    print(f"Total:{total}\tSuccess:{success_count}\tFail:{fail_count}\tSkip:{skip_count}\texists:{exists}\t{round((success_count+exists)/total, 2)}")
                continue

            if download_file(url, img_path, max_attempts=args.max_attempts, max_wait_time=args.max_wait_time):
                fail_count += 1
                if total % freq == 0:
                    print(f"Total:{total}\tSuccess:{success_count}\tFail:{fail_count}\tSkip:{skip_count}\texists:{exists}\t{round((success_count+exists)/total, 2)}")
                os.makedirs(error_save_dir, exist_ok=True)
                if not os.path.exists(error_save_path):
                    with open(error_save_path, 'w')as f:
                        json.dump(dict(row), f)
                if total % 1000==0:
                    print(f"Total:{total}\tSuccess:{success_count}\tFail:{fail_count}\tSkip:{skip_count}\tExists:{exists}\t{round((success_count+exists)/total, 2)}")
                continue
            
            with open(txt_path, 'w')as f:
                f.write(caption)
            with open(json_path, 'w')as f:
                json.dump(dict(row), f)
            success_count += 1
            if total % 1000==0:
                print(f"Total:{total}\tSuccess:{success_count}\tFail:{fail_count}\tSkip:{skip_count}\tExists:{exists}\t{round((success_count+exists)/total, 2)}")


def get_json(i, json_paths, tmp_dir=None):
    os.makedirs(tmp_dir, exist_ok=True)
    save_dir_key = [
        "data-50m-20240402",
        "data-50m-20240402-embeds/antelopev2_embeds",
    ]
    result = []
    count = 0
    total = 0
    non_exists_num = 0
    index = 0
    for json_path in tqdm(json_paths):
        total += 1

        img_path = json_path.replace(save_dir_key[1], save_dir_key[0]).replace('.json', '.jpg')
        txt_path = json_path.replace(save_dir_key[1], save_dir_key[0]).replace('.json', '.txt')
        npy_path = json_path.replace('.json', '.npy')
        
        # txt_path  = img_path.replace('.jpg', '.txt')
        # npy_path = img_path.replace(save_dir_key[0], save_dir_key[1]).replace('.jpg', '.npy')
        # json_path = img_path.replace(save_dir_key[0], save_dir_key[1]).replace('.jpg', '.json')

        # assert os.path.exists(txt_path), ValueError(f'txt path is not exists ==> {txt_path}')
        if not (os.path.exists(npy_path) and os.path.exists(json_path) and os.path.exists(txt_path) and os.path.exists(img_path)):
            non_exists_num += 1
            # if total%1000 == 0:
            #     print(f"{total-non_exists_num}/{non_exists_num}/{total}/process{i}")
            continue

        with open(txt_path, 'r')as f:
            prompt = f.readlines()[0]

        if '/home/public' in img_path:
            img_path = img_path.replace('/home/public', '/mnt/nfs/file_server/public')
            npy_path = npy_path.replace('/home/public', '/mnt/nfs/file_server/public')
            json_path = json_path.replace('/home/public', '/mnt/nfs/file_server/public')

        meta_data = {
            "image_file": img_path,
            "text": prompt,
            "embeds_path": npy_path,
            "face_info_json": json_path
        }
        result.append(meta_data)
        # if total%1000 == 0:
        #         print(f"{total-non_exists_num}/{non_exists_num}/{total}/process{i}")


        # if len(result) % 5000 == 0:
        #     print(sys.getsizeof(result) / float((1024 ** 3))) 
        if len(result) % 10000 == 0:
            tmp_path_ = os.path.join(tmp_dir, f"{i}_{index}.json")
            with open(tmp_path_, 'w')as f:
                json.dump(result, f)
            count += len(result)
            index += 1
            result = []


    tmp_path_ = os.path.join(tmp_dir, f"{i}_{index}.json")
    with open(tmp_path_, 'w')as f:
        json.dump(result, f)
    count += len(result)

    # tmp_path = os.path.join(tmp_dir, f"{i}.json")
    # with open(tmp_path, 'w')as f:
    #     json.dump(result, f)

    print(f"Process {i} finish!\tnum:{count}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_num", type=int, default=1)
    parser.add_argument("--min_reso", type=int, default=512)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--max_wait_time", type=int, default=5)
    parser.add_argument("--mode", type=str, required=True, help='d, ')
    parser.add_argument("--if_model", type=str, default='antelopev2')
    parser.add_argument("--img_paths_json", type=str, default=None, 
        help="/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/data-50m-20240402-embeds/_tmp/load_img_paths/total_img.json")
    args = parser.parse_args()

    #### download 
    if args.mode=='download':
        parquet_list = []
        data_dir = f"{source_dir}/laion_face_meta"
        parquet_list = [os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith('.parquet')]
        parquet_list = sorted(parquet_list)
        data_index = 0
        processors = []
        chunk_num = len(parquet_list) // args.process_num
        residue_num = len(parquet_list) % args.process_num
        for i in range(args.process_num):
            end_index = data_index+chunk_num+1 if i < residue_num else data_index+chunk_num
            chunk_parquet = parquet_list[data_index:end_index]
            data_index = end_index
            processor = multiprocessing.Process(
                target=download_process, 
                args=(i, chunk_parquet, args))
            processors.append(processor)
            processor.start()
        for processor in processors:
            processor.join()
    # **************************************


    #### statistic laion data num:
    elif args.mode=='statistic':
        total_num = 0
        parquet_list = []
        # source_dir = "/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/data/laion_face_data-50m"
        # data_dirs = [os.path.join(source_dir, name)for name in os.listdir(source_dir) if 'split' in name]

        data_dirs = [f"{source_dir}/laion_face_meta"]
        for data_dir in data_dirs:
            parquet_list += [os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith('.parquet')]
        for parquet_file in tqdm(parquet_list):
            pf = ParquetFile(parquet_file)
            dF = pf.to_pandas()
            total_num += len(dF)
            print(f"total: {total_num}")
    # **************************************


    #### face detect
    elif args.mode == 'face_detect':
        source_path = f"{source_dir}/data-50m-20240402"
        if args.img_paths_json is None:
            img_paths = load_paths(source_path, endswith='.jpg', process_num=2)
        else:
            with open(args.img_paths_json, 'r')as f:
                img_paths = json.load(f)
        print(f"total num:{len(img_paths)}")
        print("初始内存使用量:", memory_usage(), "MB")
        random.shuffle(img_paths)

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
            processor = multiprocessing.Process(target=face_dect, args=(i, chunk_data, args.if_model))
            processors.append(processor)
            processor.start()
        for processor in processors:
            processor.join()
    # *****************************************


    #### get train json
    elif args.mode == 'get_json':
        source_path = f"{source_dir}/data-50m-20240402-embeds/antelopev2_embeds"
        tmp_dir = f"{source_dir}/data-50m-20240402-embeds/_tmp/get_train_json"
        json_paths = load_paths(source_path, endswith='.json')
        print("初始内存使用量:", memory_usage(), "MB")

        processors = []
        chunk_num = len(json_paths) // args.process_num
        residue_num = len(json_paths) % args.process_num
        data_index = 0
        for i in range(args.process_num):
            if i < residue_num:
                chunk_data = json_paths[data_index:data_index+chunk_num+1]
                data_index = data_index+chunk_num+1
            else:
                chunk_data = json_paths[data_index:data_index+chunk_num]
                data_index = data_index+chunk_num
            processor = multiprocessing.Process(target=get_json, args=(i, chunk_data, tmp_dir))
            processors.append(processor)
            processor.start()
        for processor in processors:
            processor.join()
        if args.mode=='get_json' and tmp_dir is not None:
            save_path = "/home/public/mingjiahui/experiments/faceid/train_json/traindata_V3_test.json"
            result = []

            tmp_paths = [os.path.join(tmp_dir, name)for name in os.listdir(tmp_dir)]
            # tmp_paths = [os.path.join(tmp_dir, f"{i}.json") for i in range(args.process_num)]

            print("summarized results......")
            for tmp_path in tqdm(tmp_paths):
                with open(tmp_path, 'r')as f:
                    result += json.load(f)
            with open(save_path, 'w')as f:
                json.dump(result, f)
            print(f"result has saved in {save_path}")
            print(f"Total num:{len(result)}")
            print("内存使用量:", memory_usage(), "MB")
    # *****************************************

    else:
        print("Parameter 'mode' is invalid")
        exit(0)
