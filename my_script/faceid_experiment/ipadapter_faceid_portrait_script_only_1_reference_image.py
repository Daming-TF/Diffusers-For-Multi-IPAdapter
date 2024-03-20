import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
import numpy as np
import cv2
import os
import sys
import argparse
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from my_script.util.util import FaceidAcquirer, image_grid
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID
from ip_adapter.attention_processor import IPAttnProcessor2_0 

class IPAdapterFaceIDCustom(IPAdapterFaceID):
    def set_num_tokens(self, num_tokens):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor2_0):
                attn_processor.num_tokens = num_tokens


def main(args):
    source_dir = r'/mnt/nfs/file_server/public/mingjiahui/models'
    base_model_path =f"{source_dir}/SG161222--Realistic_Vision_V4.0_noVAE/"   # "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = f"{source_dir}/stabilityai--sd-vae-ft-mse/"   # "stabilityai/sd-vae-ft-mse"
    ip_ckpt = f"{source_dir}/h94--IP-Adapter/h94--IP-Adapter/models/ip-adapter-faceid-portrait_sd15.bin"
    device = "cuda"

    image_paths = [os.path.join(args.input_dir, name) for name in os.listdir(args.input_dir) if name.split('.')[1] in ['jpg', 'png', 'jpeg', 'webp']]

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )
    
    app = FaceidAcquirer()

    # load ip-adapter
    ip_model = IPAdapterFaceIDCustom(pipe, ip_ckpt, device, num_tokens=16, n_cond=1)

    result = None
    for image_path in image_paths:
        print(f"image_path:{image_path}")
        faceid_embeds = app.get_multi_embeds(image_path)
        suffix = os.path.basename(image_path).split('.')[1]
        txt_path = image_path.replace(suffix, 'txt')
        with open(txt_path)as f:
            prompt = f.readlines()[0]
        images = ip_model.generate(
            prompt=prompt, 
            # negative_prompt=negative_prompt, 
            faceid_embeds=faceid_embeds, 
            num_samples=4, 
            width=512, 
            height=512, 
            num_inference_steps=30, 
            seed=2023
        )
        grid = image_grid(images, 2, 2)
        save_path = os.path.join(args.save_path, os.path.basename(image_path))
        grid.save(save_path)
        print(f"result has saved in {save_path}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()
    args.save_path = args.input_dir+'_protrait_out' if args.save_path is None else args.save_path
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
