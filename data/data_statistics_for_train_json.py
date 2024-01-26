import json
import argparse
import random 
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import cv2
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt


def concat_tmp_json(args):
    result = {}
    total_num = 0
    for i in tqdm(range(args.process_num)):
        json_path = os.path.join(os.path.dirname(args.save_path), '_tmp', f"{i}.json")
        with open(json_path, 'r')as f:
            data = json.load(f)
            for k, v in data.items():
                result.setdefault(k, [])
                result[k] += v
                total_num += len(v)
    
    del_list = []
    for key in result.keys():
        if int(key) < args.reso_th:
            del_list.append(key)
    for key in del_list:
        result.pop(key)
    meta_data = []
    for result_ in result.values():
        meta_data += result_
    print(f"total num: {len(meta_data)}")

    with open(args.concat_save_path, 'w')as f:
        json.dump(meta_data, f)
    print(f"result has saved in {args.concat_save_path}")


def data_statis_processor(input_tuple):
    processor_id, meta_data_list, save_dir = input_tuple
    reso_np = np.arange(64, 512+64, 64).tolist()     # [64,128,192,256,320,384,448,512]
    result = {}
    for reso in reso_np:
        result.setdefault(reso, [])
    # result = np.zeros_like(reso_np)
    process_bar = tqdm(meta_data_list) if processor_id == 0 or processor_id == 7 else meta_data_list
    for meta_data in process_bar:
        image_file = meta_data['image_file']
        image = Image.open(image_file)
        min_reso = min(image.size)
        meta_data['crop_reso'] = min_reso

        find=False
        for reso in reso_np:
            if min_reso < reso:
                find=True
                result[reso].append(meta_data)
                break
        
        if not find:
            result[512].append(meta_data)
    
    save_path = os.path.join(save_dir, '_tmp', f"{processor_id}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w')as f:
        json.dump(result, f)
        print(f"result has saved in {save_path}")


def data_statis(args):
    print("loading json......")
    assert os.path.exists(args.input_json), ValueError(f"input json path is not exosts ==> {args.input_json}")
    with open(args.input_json, 'r')as f:
        data = json.load(f)
        print(f"json data num:>>{len(data)}<<")
    print("finish!")
    chunk_num = len(data) // args.process_num 
    residue_num = len(data) % args.process_num
    processor_input = []
    data_index = 0
    for i in range(args.process_num):
        if i < residue_num:
            chunk_input = data[data_index: data_index+chunk_num+1]
            data_index += chunk_num+1
        else:
            chunk_input = data[data_index: data_index+chunk_num]
            data_index += chunk_num
        processor_input.append((i, chunk_input, os.path.dirname(args.save_path)))
    
    with multiprocessing.Pool(processes=args.process_num)as pool:
        results = pool.map(data_statis_processor, processor_input)
        pool.close()
        pool.join()
    
    # total_result = np.zeros_like(results[0])
    # for result in results:
    #     total_result += result
    result = {}
    total_num = 0
    for i in range(args.process_num):
        json_path = os.path.join(os.path.dirname(args.save_path), '_tmp', f'{i}.json')
        with open(json_path, 'r')as f:
            data = json.load(f)
            for k, v in data.items():
                result.setdefault(k, [])
                result[k] += v
                total_num += len(v)
    total_result = [len(v) for _, v in result.items()]
    print(f"Total num:{total_num}\t{total_result}")

    x_labels = ['<64', '64-128', '128-192', '192-256', '256-320', '320-384', '384-448', '448-512']
    plt.bar(x_labels, total_result)
    save_path = os.path.join(os.path.dirname(args.save_path), 'bar_'+os.path.basename(args.save_path))
    plt.savefig(save_path)



def check_images(args):
    transform = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512)])
    print("loading json......")
    assert os.path.exists(args.input_json), ValueError(f"input json path is not exosts ==> {args.input_json}")
    with open(args.input_json, 'r')as f:
        data = json.load(f)
    print("finish!")
    # random.shuffle(data)
    meta_data_list = data[:args.check_num]
    row = int(np.ceil(args.check_num ** 0.5))
    col = args.check_num // row
    print(f"row:{row}\tcol:{col}")
    assert row*col==args.check_num and row==col, ValueError(f"The root of >>check num:{args.check_num}<< is not an int")

    image_list = []
    for meta_data in meta_data_list:
        image_file = meta_data['image_file']
        embeds_path = meta_data['embeds_path']
        reso = int(meta_data['crop_reso'])

        image = Image.open(image_file)
        h, w = image.size
        # assert reso == min(h, w), ValueError(f"The param of reso:>>{reso}<<, and min size:>>{min(h, w)}<<  ")
        assert os.path.exists(embeds_path)

        image_list.append(image)
    
    result = None
    print(f"check num:{len(meta_data_list)}")
    for i in tqdm(range(row)):
        hconcat = None
        for j in range(col):
            image = transform(image_list[row*i+j])
            hconcat = cv2.hconcat([hconcat, np.array(image)]) if hconcat is not None else np.array(image)
        result = cv2.vconcat([result, hconcat]) if result is not None else hconcat
    
        Image.fromarray(result).save(args.save_path)
    print(f"result has saved in {args.save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default='')
    parser.add_argument("--process_num", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="./data/other/check_train_data.jpg")
    parser.add_argument("--mode", type=str, required=True, help="Union[ 'ststis', 'check', 'concat']")
    # for 'check' mode
    parser.add_argument("--check_num", type=int, default=64)
    # for 'concat' mode
    parser.add_argument("--reso_th",type=int, default=256)
    parser.add_argument("--concat_save_path",type=str, default='./data/other/_tmp/result.json')
    
    
    args = parser.parse_args()
    
    if args.mode == 'ststis':
        data_statis(args)
    elif args.mode == 'check':
        check_images(args)
    elif args.mode == 'concat':
        concat_tmp_json(args)
    else:
        ValueError(f"not support this mode ==> {args.mode}")