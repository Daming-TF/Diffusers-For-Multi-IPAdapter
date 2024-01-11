
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
import argparse

import os


def main():
    source_dir = r'/mnt/nfs/file_server/public/mingjiahui/models'
    base_model_path = fr'{source_dir}/Lykon--DreamShaper/'      # "SG161222/Realistic_Vision_V4.0_noVAE"
    lora = f"{source_dir}/lora_1.5/Paper_Cutout.safetensors"
    device = "cuda"

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        feature_extractor=None,
        safety_checker=None
    ).to(device)
    pipe.load_lora_weights(
        os.path.dirname(lora), 
        weight_name=os.path.basename(lora), 
        adapter_name="style"
        )
    pipe.set_adapters(["style"], adapter_weights=[1.0])
    pipe.delete_adapters("style")
    pipe.load_lora_weights(
        os.path.dirname(lora), 
        weight_name=os.path.basename(lora), 
        adapter_name="style"
        )


    generator = torch.Generator(device).manual_seed(42)
    image = pipe(
            prompt='1 chicken',
            width=512, height=512,
            num_inference_steps=20,
            generator=generator,
        ).images[0]
    image.save('/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/debug.jpg')


if __name__ == '__main__':
    main()

"""
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
pipeline.delete_adapters("pixel")
pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
generator = torch.Generator("cuda").manual_seed(42)
image = pipeline('1 chicken').images[0]
image.save('./debug.jpg')
"""