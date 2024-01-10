import torch
from PIL import Image
import cv2
import numpy as np
import argparse
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from my_script.util import get_face_embeds, image_grid
# from my_script.unetfix import CostomUNet2DConditionModel
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from ip_adapter.attention_processor_faceid import LoRAAttnProcessor


def main(args):
    v2 = args.v2
    source_dir = r'/mnt/nfs/file_server/public/mingjiahui/models'
    base_model_path = fr'{source_dir}/Lykon--DreamShaper/'      # "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = fr"{source_dir}/stabilityai--sd-vae-ft-mse"
    image_encoder_path = fr"{source_dir}/laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
    ip_ckpt = fr"{source_dir}/h94--IP-Adapter/faceid/ip-adapter-faceid-plus_sd15.bin" if not v2 \
        else fr"{source_dir}/h94--IP-Adapter/faceid/ip-adapter-faceid-plusv2_sd15.bin"
    device = "cuda"
    print(f"\033[91m {ip_ckpt} \033[0m")

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
    print("loading vae......")
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    print("loading sdxl......")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        # unet=unet,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )

    # load ip-adapter
    ip_model = IPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device)

    # prepare data
    image_dirs = [os.path.join(args.input, name) for name in os.listdir(args.input)]
    print(image_dirs)
    image_paths = []
    for image_dir in image_dirs:
        image_paths += [os.path.join(image_dir, name) for name in os.listdir(image_dir) \
                        if name.split('.')[1] in ['jpg', 'png', 'webp']]

    for image_path in image_paths:
        # 1.get face imbeds
        faceid_embeds, face_image = get_face_embeds(cv2.imread(image_path))
        # 2.get prompt
        suffix = os.path.basename(image_path).split('.')[1]
        txt_path = image_path.replace(suffix, 'txt')
        print(suffix, txt_path, image_path)
        with open(txt_path, 'r')as f:
            txt_content = f.readlines()
        assert len(txt_content)==1
        prompt = txt_content[0]
        # 3.processing
        images = ip_model.generate(
            prompt=prompt, face_image=face_image, faceid_embeds=faceid_embeds, shortcut=v2, s_scale=args.s_scale,
            num_samples=4, width=512, height=512, num_inference_steps=30, seed=42, guidance_scale=6,
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
    parser.add_argument("--input", type=str, required=True, help="input image dir")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--v2", action='store_true')
    parser.add_argument("--s_scale", type=float, default=1.0)
    args = parser.parse_args()
    args.output = args.input + '_sd15_plusv2_script_output' \
        if args.output is None else args.output
    os.makedirs(args.output, exist_ok=True)
    print(f"V2:{args.v2}")
    print(f"result will saved in {args.output}")
    main(args)