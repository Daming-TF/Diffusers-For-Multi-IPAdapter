import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
import requests
import argparse
import multiprocessing
import math


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
            txt_path = os.path.join(save_dir_, f"{j}.txt")
            json_path = os.path.join(save_dir_, f"{j}.json")
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

    urls, description, color, category, sku = load_csv(args.csv_file)
    assert len(urls)==len(description)==len(color)==len(category)==len(sku), ValueError("some error happened")

    data_index = 0
    processors = []
    chunk_num = len(urls) // args.process_num
    residue_num = len(urls) % args.process_num
    for i in range(args.process_num):
        end_index = data_index+chunk_num+1 if i < residue_num else data_index+chunk_num

        chunk_urls = urls[data_index:end_index]
        chunk_description = description[data_index:end_index]
        chunk_color = color[data_index:end_index]
        chunk_category = category[data_index:end_index]
        chunk_sku = sku[data_index:end_index]

        processor = multiprocessing.Process(
            target=download_process, 
            args=(i, chunk_urls, chunk_description, chunk_color, chunk_category, chunk_sku, args))
        processors.append(processor)
        processor.start()
    for processor in processors:
        processor.join()

