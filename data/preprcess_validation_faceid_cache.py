
from insightface.app import FaceAnalysis
import os
from PIL import Image
import numpy as np
import cv2
from copy import deepcopy
from torchvision import transforms
import torch
import json
from tqdm import tqdm

input_dirs = [
    "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/famous",
    "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/average_id"
]
img_paths = []
for input_dir in input_dirs:
    img_paths += [os.path.join(input_dir, name)for name in os.listdir(input_dir) if name.split('.')[-1] not in ['txt', 'npy', 'json']]
app0 = FaceAnalysis(name='/home/mingjiahui/.insightface/models/antelopev2/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app1 = FaceAnalysis(name='/home/mingjiahui/.insightface/models/buffalo_l/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app0.prepare(ctx_id=0, det_size=(640, 640))
app1.prepare(ctx_id=0, det_size=(640, 640))
transfer = transforms.Resize(768)

meta_template = {}
meta_template.setdefault('landmark_2d_106', [])
meta_template.setdefault('landmark_3d_68', [])
meta_template.setdefault('kps', [])
meta_template.setdefault('pose', [])
meta_template.setdefault('sex', [])
meta_template.setdefault('age', [])
meta_template.setdefault('bbox', [])
meta_template.setdefault('antelopev2_embeds', [])
meta_template.setdefault('buffalo_l_embeds', [])

for img_path in tqdm(img_paths):
    face_image = Image.open(img_path).convert("RGB")
    ori_w, _ = face_image.size
    face_image = transfer(face_image)
    new_w, _ = face_image.size
    ratio = ori_w / new_w
    face_info_list0 = app0.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info_list1 = app1.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    if len(face_info_list0)==0 or len(face_info_list1)==0:
        continue
    meta_data = deepcopy(meta_template)

    ## M1
    face_info = sorted(face_info_list0, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    meta_data['image_file'] = img_path
    meta_data['landmark_2d_106'].append((face_info["landmark_2d_106"]*ratio).tolist())
    meta_data['landmark_3d_68'].append((face_info["landmark_3d_68"]*ratio).tolist())
    meta_data['kps'].append((face_info["kps"]*ratio).tolist())
    meta_data['pose'].append((face_info["pose"]*ratio).tolist())
    meta_data['sex'].append(face_info.sex)
    meta_data['age'].append(face_info["age"])
    meta_data['bbox'].append((face_info["bbox"]*ratio).tolist())
    face_emb = torch.from_numpy(face_info.embedding).unsqueeze(0)
    cache_save_dir = os.path.dirname(img_path)
    save_name = os.path.basename(img_path).split('.')[0]
    cache_path = os.path.join(cache_save_dir, save_name+'--antelopev2.npy')
    np.save(cache_path, face_emb)
    meta_data['antelopev2_embeds'].append(cache_path)
    ## M2
    face_info = sorted(face_info_list1, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
    face_emb = torch.from_numpy(face_info.embedding).unsqueeze(0)
    cache_save_dir = os.path.dirname(img_path)
    save_name = os.path.basename(img_path).split('.')[0]
    cache_path = os.path.join(cache_save_dir, save_name+'--buffalo_l.npy')
    np.save(cache_path, face_emb)
    meta_data['buffalo_l_embeds'].append(cache_path)
    
    json_path = os.path.join(cache_save_dir, save_name+'.json')
    with open(json_path, 'w')as f:
        json.dump(meta_data, f)  
    print(f"result has saved in {json_path}")
