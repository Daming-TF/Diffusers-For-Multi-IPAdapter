# prompt_template= "a young woman with a {}, wearing a pink shirt. She is standing in front of a fence, possibly in a park or an outdoor setting. The woman appears to be enjoying her time outdoors, possibly engaging in a sport or a recreational activity. "
# expression_leys = ['bright smile', 'sad face', 'astonished face', 'exaggerated expression']
# for expression_ley in expression_leys:
#     print(prompt_template.format(expression_ley))
#     print('\n')

# print(all([1, 1]))

# import torch

# ckpt = "/mnt/nfs/file_server/public/mingjiahui/models/InstantX--InstantID/ip-adapter.bin" 
# sd = torch.load(ckpt) 
# print(sd.keys())


import multiprocessing
import json
from PIL import Image
from tqdm import tqdm
import os
import random


def processing(data, i, save_dir):
    result = []
    save_path = os.path.join(save_dir, f"{i}.json")
    for data_ in tqdm(data):
        image_path = data_['image_file']
        image = Image.open(image_path)
        if min(image.size) < 768:
            continue
        result.append(data_)
        # break
    with open(save_path, 'w')as f:
        json.dump(result, f)
    print(f"result has saved in {save_path}")


if __name__ == '__main__':
    json_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_with_all_face_info.json"
    save_dir = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/_tmp/min_reso_512"
    os.makedirs(save_dir, exist_ok=True)
    save_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_with_all_face_info--min_reso_768.json"
    result = []
    process_num = 4
    processors = []
    
    with open(json_path, 'r')as f:
        data = json.load(f)
        random.shuffle(data)
        
    print(f"total data num:{len(data)}")
    data_index = 0
    chunk_num = len(data) // process_num
    residue_num = len(data) % process_num
    for i in range(process_num):
        if i < residue_num:
            chunk_data = data[data_index:data_index+chunk_num+1]
            data_index = data_index+chunk_num+1
        else:
            chunk_data = data[data_index:data_index+chunk_num]
            data_index = data_index+chunk_num
        processor = multiprocessing.Process(target=processing, args=(chunk_data, i, save_dir))
        processors.append(processor)
        processor.start()
    for processor in processors:
        processor.join()

    for i in range(process_num):
        json_path = os.path.join(save_dir, f"{i}.json")
        with open(json_path, 'r')as f:
            result_ = json.load(f)
            result += result_
    with open(save_path, 'w')as f:
        json.dump(result, f)
    print(f"result has save in {save_path}\t Total num:{len(result)}")
