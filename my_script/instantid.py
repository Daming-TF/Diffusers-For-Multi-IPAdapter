import cv2
import torch
import numpy as np
from PIL import Image
import os
import argparse

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from InstantID.pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from InstantID.infer import resize_img
from my_script.util.util import image_grid


def main(args):
    id_path = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/test/mjh_cry.jpg"
    id1_path = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/test/mjh_laughing.jpg"
    kps_path = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/test/2.png"

    app = FaceAnalysis(name='/home/mingjiahui/.insightface/models/antelopev2/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
     # 1.get face embeds
    face_image = Image.open(id_path).convert("RGB")
    face_image = resize_img(face_image)
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]   # only use the maximum face
    face_emb = face_info['embedding']
    face_image = Image.open(id1_path).convert("RGB")
    face_image = resize_img(face_image)
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]   # only use the maximum face
    face_emb1 = face_info['embedding']
    face_emb = np.mean([face_emb, face_emb1], axis=0)
    
    # 2. get face kps
    face_image = Image.open(kps_path).convert("RGB")
    face_image = resize_img(face_image)
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]   # only use the maximum face
    face_kps = draw_kps(face_image, face_info['kps'])

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

    suffix = os.path.basename(id_path).split('.')[1]
    txt_path = id_path.replace(suffix, 'txt')
    # with open(txt_path, 'r')as f:
    #     prompt = f.readlines()[0]
    #  with a sad face
    #  broken-hearted 
    prompt = "The image features a young boy wearing a black shirt. He is standing in front of a fence. The background includes a bright light, possibly from a nearby street light or a light source in the distance."
    
   
    pipe.set_ip_adapter_scale(0.8)
    images = []
    for _ in range(args.batch):
        generator = torch.Generator('cuda').manual_seed(42+_)
        image = pipe(
            prompt=prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=0.8,
            num_inference_steps=30,
            guidance_scale=5,
            # num_images_per_prompt=4,
            generator=generator
        ).images[0]
        images.append(image)
    reslut = image_grid(images, 2, 2)
    save_path = os.path.join(args.save_dir, f"id-{os.path.basename(id_path).split('.')[0]}--kps-{os.path.basename(kps_path).split('.')[0]}.jpg")
    reslut.save(save_path)
    print(f"result has saved in ==> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_dir", type=str, default="/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data")
    parser.add_argument("--save_dir", type=str, default="/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/InstantID/script_output")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lora", type=str, default=None)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
    