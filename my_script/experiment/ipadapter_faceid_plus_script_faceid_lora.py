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
from my_script.ipadapter_faceid_plus import CostomIPAdapterFaceIDPlus
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from ip_adapter.attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor


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
    ip_model = CostomIPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device)

    # generate image
    prompt = "closeup photo of a man wearing a white shirt in a garden, high quality, diffuse light, highly detailed, 4k"
    negative_prompt = "blurry, malformed, distorted, naked"

    # jiahui's modify
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(512)
    ])
    image_path = args.input
    image = transform(Image.open(image_path).convert("RGB"))
    faceid_embeds, face_image = get_face_embeds(image)

    result = None
    start, end, interval = args.faceid_lora_weight.split('-')
    faceid_lora_weights = np.arange(float(start), float(end)+float(interval), float(interval))
    for faceid_lora_weight in faceid_lora_weights:
        ip_model.set_lora_scale(faceid_lora_weight)
        images = ip_model.generate(
            prompt=prompt, negative_prompt=negative_prompt, face_image=face_image, faceid_embeds=faceid_embeds, shortcut=v2, s_scale=1.0,
            num_samples=4, width=512, height=512, num_inference_steps=30, seed=42, guidance_scale=6,
        )
        grid = np.array(image_grid(images, 2, 2))
        result = cv2.hconcat([result, grid]) if result is not None else grid

    save_name = f"{os.path.basename(ip_ckpt).split('.')[0]}"
    save_name += f"-{os.path.basename(args.input).split('.')[0]}"
    save_path = os.path.join(args.output, save_name+'.jpg')
    Image.fromarray(result).save(save_path)
    print(f"result has saved in {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input image path")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--faceid_lora_weight", type=str, default="0-1-0.2")
    parser.add_argument("--v2", action='store_true')
    args = parser.parse_args()
    args.output = os.path.dirname(args.input) + '_output' \
        if args.output is None else args.output
    os.makedirs(args.output, exist_ok=True)
    print(args.output)
    main(args)