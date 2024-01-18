import cv2
from PIL import Image
from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import argparse

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from my_script.util.util import image_grid, FaceidAcquirer
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL


def main(args):
    v2 = True
    source_dir = "/mnt/nfs/file_server/public/mingjiahui/models"
    base_model_path = "/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/"     # "SG161222/Realistic_Vision_V4.0_noVAE"
    # vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path =  f"{source_dir}/laion--CLIP-ViT-H-14-laion2B-s32B-b79K"      # "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    ip_ckpt = f"{source_dir}/h94--IP-Adapter/h94--IP-Adapter/sdxl_models/ip-adapter-faceid-plusv2_sdxl.bin"
    assert 'v2' in ip_ckpt, "sdxl only support plusv2, not supoort faceid plus"
    device = "cuda"
    app = FaceidAcquirer()

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    # vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        # vae=vae,
        add_watermarker=False,
    )

    # load ip-adapter
    ip_model = IPAdapterFaceIDPlusXL(pipe, image_encoder_path, ip_ckpt, device)

    # prepare promot
    suffix = os.path.basename(args.input).split('.')[1]
    txt_path = args.input.replace(suffix, 'txt')
    if os.path.exists(txt_path):
        with open(txt_path, 'r')as f:
            data = f.readlines()
            assert len(data)==1
        args.prompt = data[0]
    prompt = "closeup photo of a man wearing a white shirt in a garden, high quality, diffuse light, highly detailed, 4k"  \
        if args.prompt is None else args.prompt
    negative_prompt = "blurry, malformed, distorted, naked" \
        if args.negative_prompt is None else args.negative_prompt
    
    # prepare face id
    faceid_embeds, face_image = app.get_face_embeds(cv2.imread(args.input))

    # generate image
    images = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, face_image=face_image, faceid_embeds=faceid_embeds, shortcut=v2, s_scale=1.0,
        num_samples=4, width=1024, height=1024, num_inference_steps=30, seed=42
    )
    grid = image_grid(images, 2, 2)
    grid.save(args.output)
    print(f"result has saved in {args.output} ......")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input image path")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)
    args = parser.parse_args()
    assert os.path.isfile(args.input)
    args.output = os.path.join(os.path.dirname(args.input)+'__faceid_plusV2_xl', os.path.basename(args.input))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f" save path: {args.output}")
    main(args)
