import os
import json
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from tqdm import tqdm
transform = transforms.Resize(1024)
check_num = 10
# input_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/coyo700m/data_arcface/00000/'
input_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/data/data-50m_arcface/00000/00000/'
save_path = r'./data/other/data_arcface_check_data.jpg'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
npy_paths = [os.path.join(input_dir, name) for name in os.listdir(input_dir) if name.endswith('.npy')][:check_num]

reuslt = None
for npy_path in tqdm(npy_paths):
    json_path = npy_path.replace('npy', 'json')
    with open(json_path, 'r')as f:
        metadata = json.load(f)
        image_path = metadata['image_path']
        bbox = metadata['bbox']

    image = np.array(transform(Image.open(image_path)))
    x,y,w,h = bbox
    cv2.rectangle(image, (int(x), int(y)), (int(w), int(h)), (0,255,0), 4)
    image = cv2.resize(image, (512, 512))
    reuslt = cv2.hconcat([reuslt, image])if reuslt is not None else image

Image.fromarray(reuslt).save(save_path)
