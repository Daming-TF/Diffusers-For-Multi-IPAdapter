import os
import json
from PIL import Image
import numpy as np
import cv2
import random
from torchvision import transforms
from tqdm import tqdm
import math
transform = transforms.Resize(1024)
check_num = 64
row_col = math.sqrt(check_num)

save_path = r'./data/other/coyo_arcface_check_data.jpg'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

json_path = "/mnt/nfs/file_server/public/mingjiahui/data/coyo700m/_tmp/train-coyo.json"
with open(json_path, 'r')as f:
    metadata_list = json.load(f)

random_metadata = random.sample(metadata_list, check_num)

result = None
for i in tqdm(range(int(row_col))):
    h_concat = None
    for j in range(int(row_col)):
        metadata = random_metadata[i*int(row_col)+j]
        image_file = metadata['image_file']
        embeds_path = metadata['embeds_path']
        json_path = embeds_path.replace('npy', 'json')
        with open(json_path, 'r')as f:
            metadata = json.load(f)
            image_path = metadata['image_path']
            bbox = metadata['bbox']
            assert image_path==image_file

        image = np.array(transform(Image.open(image_path)))
        x,y,w,h = bbox
        cv2.rectangle(image, (int(x), int(y)), (int(w), int(h)), (0,255,0), 4)
        image = cv2.resize(image, (256, 256))
        h_concat = cv2.hconcat([h_concat, image]) if h_concat is not None else image
    result = cv2.vconcat([result, h_concat]) if result is not None else h_concat

Image.fromarray(result).save(save_path)


# input_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/coyo700m/data_arcface/00000/'
# input_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/data/data-50m_arcface/00000/00000/'

# npy_paths = [os.path.join(input_dir, name) for name in os.listdir(input_dir) if name.endswith('.npy')][:check_num]

# reuslt = None
# for npy_path in tqdm(npy_paths):
#     json_path = npy_path.replace('npy', 'json')
#     with open(json_path, 'r')as f:
#         metadata = json.load(f)
#         image_path = metadata['image_path']
#         bbox = metadata['bbox']

#     image = np.array(transform(Image.open(image_path)))
#     x,y,w,h = bbox
#     cv2.rectangle(image, (int(x), int(y)), (int(w), int(h)), (0,255,0), 4)
#     image = cv2.resize(image, (512, 512))
#     reuslt = cv2.hconcat([reuslt, image])if reuslt is not None else image

# Image.fromarray(reuslt).save(save_path)
