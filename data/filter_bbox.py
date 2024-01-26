"""
1.input train json file
2.get json file for per image
3.get boundbox info
4.filter smaller boundbox
V1: /mnt/nfs/file_server/public/mingjiahui/experiments/faceid/tran_json/traindata_V1.json
"""
import json
import argparse
import multiprocessing
from tqdm import tqdm
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import random


data_dict = {
    'Laion': ['data-50m', 'data_50m-crop_V1'],
    'coyo': ['coyo700m/data', 'coyo700m/data_crop_V1'],
    'ffhq': ['in-the-wild-images', 'in-the-wild-images_crop_V1'],
}
tmp = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/_tmp"
transform = transforms.Resize(1024)


def crop_image(image_path, bbox, factor=2):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(e)
        return None, None
    w, h = image.size
    min_size = min(w, h)
    resize_ratio = min_size /1024
    image = transform(image)
    w, h = image.size
    l_top_x, l_top_y, r_bottom_x, r_bottom_y = bbox
    l_top_x = max(0, l_top_x)
    l_top_y = max(0, l_top_y)
    r_bottom_x = min(w, r_bottom_x)
    r_bottom_y = min(h, r_bottom_y)

    b_w, b_h = (r_bottom_x - l_top_x), (r_bottom_y - l_top_y)
    center_x = l_top_x + b_w//2
    center_y = l_top_y + b_h//2

    # new_size = min(min(b_w, b_h) * factor, min(w, h))
    new_size = max(b_w, b_h) * factor
    x_start = max(0, center_x-new_size//2)
    y_start = max(0, center_y-new_size//2)
    x_end = min(w, center_x+new_size//2)
    y_end = min(h, center_y+new_size//2)
    if x_start==0 or y_start==0:
        x_end = min(w, x_start+new_size)
        y_end = min(h, y_start+new_size)
    elif x_end==w or y_end==h:
        x_start = max(0, x_end-new_size)
        y_start = max(0, y_end-new_size)
    ratio = (x_end - x_start) / (y_end - y_start)

    # if ratio != 1:
    #     print(f"{ratio}: {image_path}\t{x_start}\t{y_start}\t{x_end}\t{y_end}")
    #     exit(0)
    #     return None, None
    # else:
    crop_image = Image.fromarray(np.array(image)[int(y_start):int(y_end), int(x_start):int(x_end)])
    new_w, new_h = crop_image.size
    crop_image = crop_image.resize((int(new_w*resize_ratio), int(new_h*resize_ratio)))
    match = False
    for data_type, replace_key in data_dict.items():
        if data_type in image_path and replace_key[0] in image_path:
            match = True
            save_path = image_path.replace(replace_key[0], replace_key[1])
            break
    assert match, ValueError("data type looks some error")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    crop_image.save(save_path)
    # if ratio != 1:
    #     print(save_path)
    #     exit(0)
    return save_path, min(crop_image.size)


def processing(tuple_input):
    (processor_id, meta_data, save_crop, check) = tuple_input
    results = []
    progress_bar = tqdm(meta_data) if processor_id == 0 or check else meta_data
    for meta_data_ in progress_bar:
        image_file = meta_data_['image_file']
        embeds_path = meta_data_['embeds_path']
        json_path = embeds_path.replace('npy', 'json')

        with open(json_path, 'r')as f:
            info = json.load(f)
            bbox = info['bbox']
        
        save_path, crop_reso = crop_image(image_file, bbox) if save_crop else (None, None)

        if save_path is not None:
            meta_data_['image_file'] = save_path
            meta_data_['crop_reso'] = crop_reso
            results.append(meta_data_)
    json_path = os.path.join(tmp, f"{processor_id}.json")
    with open(json_path, 'w')as f:
        json.dump(results, f)
        print(f"tmp json has saved in {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--processor_num", type=int, default=1)
    parser.add_argument("--save_crop", action='store_true')
    args = parser.parse_args()
    args.save_path = os.path.join(os.path.dirname(args.json_path), os.path.basename(args.json_path).split('.')[0]+'_crop.json') if args.save_path is None \
        else args.save_path
    print(f"save path :{args.save_path}")

    with open(args.json_path, 'r')as f:
        metadata = json.load(f)
        random.shuffle(metadata)

    chunk_num = len(metadata) // args.processor_num
    residue_num = len(metadata) % args.processor_num
    multi_process_input = []
    data_index = 0
    for i in range(args.processor_num):
        if i < residue_num:
            chunk_input = metadata[data_index: data_index+chunk_num+1]
            data_index = data_index+chunk_num+1
        else:
            chunk_input = metadata[data_index: data_index+chunk_num]
            data_index = data_index+chunk_num
        check = True if i == args.processor_num -1 else False
        multi_process_input.append((i, chunk_input, args.save_crop, check))
    
    with multiprocessing.Pool(processes=args.processor_num)as pool:
        results = pool.map(processing, multi_process_input)
        # results = [pool.apply_async(processing, (input_data,)) for input_data in multi_process_input]
        pool.close()
        pool.join()
        

    # save
    print("loadding tmp json......")
    result = []
    for i in tqdm(range(args.processor_num)):
        json_path = os.path.join(tmp, f"{i}.json")
        with open(json_path, 'r')as f:
            data = json.load(f)
            result += data
        os.remove(json_path)
    # result = sorted(result)
    with open(args.save_path, 'w')as f:
        json.dump(result, f)
    print(f"result has saved in {args.save_path}")
