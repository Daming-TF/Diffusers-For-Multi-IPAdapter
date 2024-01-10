import torch
from PIL import Image
import cv2
import numpy as np
import argparse
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from my_script.util import get_face_embeds, image_grid
# from my_script.unetfix import CostomUNet2DConditionModel
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL
from my_script.ipadapter_faceid_xl import CostomIPAdapterFaceIDXL


def main(args):
    base_model_path = r"/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/"       # "SG161222/RealVisXL_V3.0"
    ip_ckpt = r"/mnt/nfs/file_server/public/mingjiahui/models/h94--IP-Adapter/faceid/ip-adapter-faceid_sdxl.bin"
    device = "cuda"

    print("loading scheduler......")
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    print("loading sdxl......")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        add_watermarker=False,
    )

    # load ip-adapter
    ip_model = CostomIPAdapterFaceIDXL(pipe, ip_ckpt, device)
    ip_model.set_lora_scale(args.ip_lora)

    # prepare data
    image_dirs = [os.path.join(args.input, name) for name in os.listdir(args.input)]
    image_paths = []
    for image_dir in image_dirs:
        image_paths += [os.path.join(image_dir, name) for name in os.listdir(image_dir) \
                        if name.split('.')[1] in ['jpg', 'png', 'webp']]

    for image_path in image_paths:
        # 1.get face imbeds
        faceid_embeds, _ = get_face_embeds(cv2.imread(image_path))
        # 2.get prompt
        suffix = os.path.basename(image_path).split('.')[1]
        txt_path = image_path.replace(suffix, 'txt')
        with open(txt_path, 'r')as f:
            txt_content = f.readlines()
        assert len(txt_content)==1
        prompt = txt_content[0]
        # 3.processing
        print(f"prompt:{prompt}\nip lora:{args.ip_lora}\tip scale:{args.ip_scale}")
        images = ip_model.generate(
            prompt=prompt, faceid_embeds=faceid_embeds, num_samples=4, \
            width=1024, height=1024, \
            num_inference_steps=30, seed=42, guidance_scale=6,
            scale=args.ip_scale
        )
        grid = image_grid(images, 2, 2)
        
        image_dir_name = os.path.basename(os.path.dirname(image_path))
        save_dir = os.path.join(args.output, image_dir_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        grid.save(save_path)
        print(f"result has saved in ==> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, 
                        default='./data/test_data_V2', help="input image dir")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--ip_scale", type=float, default=1.0)
    parser.add_argument("--ip_lora", type=float, default=1.0)
    args = parser.parse_args()
    args.output = args.input + f'_xl_script_scale{args.ip_scale}-ip_lora{args.ip_lora}_output' \
        if args.output is None else args.output
    os.makedirs(args.output, exist_ok=True)
    print(args.output)
    main(args)
    