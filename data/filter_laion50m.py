from fastparquet import ParquetFile
import os
from tqdm import tqdm
import requests
import json
import argparse
import multiprocessing
from requests.exceptions import Timeout
import hashlib


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
    for parquet_file in tqdm(parquet_list):
        success_count = 0
        fail_count = 0
        skip_count = 0
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
                # print(parquet_file)
                print(e)
                continue

            save_dir = parquet_file.replace(save_key[0], save_key[1]).replace('.parquet', '')
            os.makedirs(save_dir, exist_ok=True)
            # save_name = str(index).zfill(8)
            img_path  = os.path.join(save_dir, hashed_pair+'.jpg')
            txt_path  = os.path.join(save_dir, hashed_pair+'.txt')
            json_path  = os.path.join(save_dir, hashed_pair+'.json')
            error_save_path = os.path.join(error_save_dir, hashed_pair+'.json')
            if (os.path.exists(img_path) and os.path.exists(txt_path) and os.path.exists(json_path)) or os.path.exists(error_save_path):
                success_count += 1
                continue

            if download_file(url, img_path, max_attempts=args.max_attempts, max_wait_time=args.max_wait_time):
                fail_count += 1
                os.makedirs(error_save_dir, exist_ok=True)
                with open(error_save_path, 'w')as f:
                    json.dump(dict(row), f)
                print(f"Total:{total}\tSuccess:{success_count}\tFail:{fail_count}\tSkip:{skip_count}\t{round(success_count/total, 2)}")
                continue
            
            with open(txt_path, 'w')as f:
                f.write(caption)
            with open(json_path, 'w')as f:
                json.dump(dict(row), f)
            success_count += 1
            if total % 1000==0:
                print(f"Total:{total}\tSuccess:{success_count}\tFail:{fail_count}\tSkip:{skip_count}\t{round(success_count/total, 2)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_num", type=int, default=1)
    parser.add_argument("--min_reso", type=int, default=512)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--max_wait_time", type=int, default=5)
    args = parser.parse_args()

    #### download 
    parquet_list = []
    data_dir = "/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/laion_face_meta"
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


    # #### statistic laion data num:
    # total_num = 0
    # parquet_list = []
    # # source_dir = "/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/data/laion_face_data-50m"
    # # data_dirs = [os.path.join(source_dir, name)for name in os.listdir(source_dir) if 'split' in name]

    # data_dirs = ["/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/laion_face_meta"]
    # for data_dir in data_dirs:
    #     parquet_list += [os.path.join(data_dir, name) for name in os.listdir(data_dir) if name.endswith('.parquet')]
    # for parquet_file in tqdm(parquet_list):
    #     pf = ParquetFile(parquet_file)
    #     dF = pf.to_pandas()
    #     total_num += len(dF)
    #     print(f"total: {total_num}")
    # # **************************************

