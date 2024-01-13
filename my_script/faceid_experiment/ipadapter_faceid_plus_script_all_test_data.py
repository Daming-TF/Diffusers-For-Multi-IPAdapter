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

    # # generate image
    # prompt = "closeup photo of a man wearing a white shirt in a garden, high quality, diffuse light, highly detailed, 4k"
    # # prompt = "closeup photo of a girl wearing a white dress in a garden, high quality, diffuse light, highly detailed, 4k"
    # negative_prompt = "blurry, malformed, distorted, naked"
    # # 'suren1', 'prof_pic_1', 'suren4', 'suren5', 'suren6', 'suren7', 'suren8'
    # keys = ['suren9.jpg','suren10.jpg','suren11.jpg', 'test.jpg']

    # jiahui's modify
    image_dir = args.input
    image_paths = [os.path.join(image_dir, name) for name in os.listdir(image_dir) \
                   if name.split('.')[1] in ['jpg', 'png']]
    # from torchvision import transforms
    # transform = transforms.Compose([
    #     transforms.Resize(1024)
    # ])

    for image_path in image_paths:
        # (1)get reference image
        # faceid_embeds, face_image = get_face_embeds(transform(Image.open(image_path).convert("RGB")))
        faceid_embeds, face_image = get_face_embeds(Image.open(image_path).convert("RGB"))
        # (2)get prompt
        suffix = os.path.basename(image_path).split('.')[1]
        txt_path = image_path.replace(suffix, 'txt')
        with open(txt_path, 'r')as f:
            prompts = f.readlines()
            assert len(prompts)==1, "txt file looks happening some error"
        prompt = prompts[0]

        images = ip_model.generate(
            prompt=prompt, 
            # negative_prompt=negative_prompt, 
            face_image=face_image, faceid_embeds=faceid_embeds, 
            shortcut=v2, s_scale=args.s_scale,
            num_samples=4, 
            width=512, height=512, 
            num_inference_steps=30, 
            seed=42, 
            guidance_scale=7.5,
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
    parser.add_argument("--input", type=str, required=True, help="input image path")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--v2", action='store_true')
    parser.add_argument("--s_scale", type=float, default=1.0)
    args = parser.parse_args()
    args.output = args.input + '_output' if args.output is None else args.output
    os.makedirs(args.output, exist_ok=True)
    print(f"V2:{args.v2}")
    print(f"result will saved in {args.output}")
    main(args)