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
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from ip_adapter.attention_processor_faceid import LoRAAttnProcessor


def main(args):
    source_dir = r'/mnt/nfs/file_server/public/mingjiahui/models'
    base_model_path = fr'{source_dir}/Lykon--DreamShaper/'      # "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = fr"{source_dir}/stabilityai--sd-vae-ft-mse"
    ip_ckpt = fr"{source_dir}/h94--IP-Adapter/faceid/ip-adapter-faceid_sd15.bin"
    # lora = f"{source_dir}/h94--IP-Adapter/faceid/ip-adapter-faceid_sd15_lora.safetensors"
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
    ip_model = IPAdapterFaceID(pipe, ip_ckpt, device)

    # generate image
    # prompt = "closeup photo of a man wearing a white shirt in a garden, high quality, diffuse light, highly detailed, 4k"
    prompt = "closeup photo of a girl wearing a white dress in a garden, high quality, diffuse light, highly detailed, 4k"
    negative_prompt = "blurry, malformed, distorted, naked"
    # 'suren1', 'prof_pic_1', 'suren4', 'suren5', 'suren6', 'suren7', 'suren8'
    keys = ['suren1.jpg']

    # jiahui's modify
    image_dir = args.input
    image_paths = [os.path.join(image_dir, name) for name in os.listdir(image_dir)]
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(1024)
    ])

    for image_path in image_paths:
        skip = True
        for key in keys:
            if key in image_path:
                skip = False
                break
        
        if skip:
            continue

        faceid_embeds, _ = get_face_embeds(transform(Image.open(image_path).convert("RGB")))
        images = ip_model.generate(
            prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=4, width=512, height=512, num_inference_steps=30, seed=42, guidance_scale=6,
        )
        grid = np.array(image_grid(images, 2, 2))

        save_name = f"{os.path.basename(ip_ckpt).split('.')[0]}"
        save_name += f"-{os.path.basename(image_path).split('.')[0]}"
        save_name += f"-lora_1"
        save_path = os.path.join(args.output, save_name+'.jpg')
        Image.fromarray(grid).save(save_path)
        print(f"result has saved in {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input image dir")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    args.output = args.input + '_output' \
        if args.output is None else args.output
    os.makedirs(args.output, exist_ok=True)
    print(f"result will save in ==> {args.output}")
    assert os.path.isdir(args.output)
    main(args)