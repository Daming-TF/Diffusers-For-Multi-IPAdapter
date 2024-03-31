import json
import random
import os
import math
import numpy as np
import cv2
from PIL import Image

import torch
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from InstantID.pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
# from InstantID.infer import resize_img
from my_script.util.util import image_grid


def resize_and_draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)], size=1024):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    # resize
    w, h = image_pil.size
    if w > h:
        new_w = size
        new_h = int(new_w/w * h)
    else:
        new_h = size
        new_w = int(new_h/h * w)
    ratio = new_h / h
    kps = kps * ratio
    
    out_img = np.zeros([new_h, new_w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


if __name__ =='__main__':
    # 1. get meta data
    json_dir = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json"
    # json_name = "traindata_V1_with_all_face_info--antelopev2_non_norm--min_reso_768.json"
    json_name = "traindata_V2_procucts_asos.json"
    save_dir = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/debug"
    save_dir = os.path.join(save_dir, os.path.basename(json_name).split('.')[0])
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(json_dir, json_name)
    with open(json_path, 'r')as f:
        data = json.load(f)
        data = random.sample(data, 16)

    # 2. prepare model
    # Path to InstantID models
    source_dir = f'/mnt/nfs/file_server/public/mingjiahui/models'
    face_adapter = f'{source_dir}/InstantX--InstantID/ip-adapter.bin'
    controlnet_path = f'{source_dir}/InstantX--InstantID/ControlNetModel'

    # Load pipeline
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, 
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    # base_model_path = '/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/'
    base_model_path="/mnt/nfs/file_server/public/mingjiahui/models/wangqixun--YamerMIX_v8/"
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.cuda()
    pipe.load_ip_adapter_instantid(face_adapter)


    # 3. processing
    for data_ in data:
        img_path = data_['image_file']
        prompt = data_['text']
        embeds_path = data_['embeds_path']
        face_info_json = data_['face_info_json']

        img = Image.open(img_path)
        with open(face_info_json, 'r')as f:
            face_info = json.load(f)
            kps = np.array(face_info['kps'])
            if len(kps.shape) == 3:
                kps = kps[0]
        face_kps = resize_and_draw_kps(img, kps)
        face_emb = np.load(embeds_path)

        ## debug
        # img = img.resize(face_kps.size)
        # debug = cv2.hconcat([np.array(img), np.array(face_kps)])
        # Image.fromarray(debug).save("/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/debug/debug.jpg")
        # print(f"result has saved in /home/mingjiahui/projects/IpAdapter/IP-Adapter/data/debug/debug.jpg")
        # break

        pipe.set_ip_adapter_scale(0.8)
        generator = torch.Generator('cuda').manual_seed(42)
        result_ = pipe(
            prompt=prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=0.8,
            num_inference_steps=30,
            guidance_scale=5,
            # num_images_per_prompt=4,
            generator=generator
        ).images[0]
        try:
            result = cv2.hconcat([
                np.array(result_), 
                np.array(img.resize(face_kps.size)), 
                np.array(face_kps)]
                )
        except Exception as e:
            print(e)
            print(result_.size, img.resize(face_kps.size).size, face_kps.size)
            continue
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        Image.fromarray(result).save(save_path)
        print(f"result has saved in {save_path}")