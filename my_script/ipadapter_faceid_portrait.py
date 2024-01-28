import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
import numpy as np
import os
import sys
import argparse
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from my_script.util.util import FaceidAcquirer, image_grid
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID


def main(args):
    source_dir = r'/mnt/nfs/file_server/public/mingjiahui/models'
    base_model_path =f"{source_dir}/SG161222--Realistic_Vision_V4.0_noVAE/"   # "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = f"{source_dir}/stabilityai--sd-vae-ft-mse/"   # "stabilityai/sd-vae-ft-mse"
    ip_ckpt = f"{source_dir}/h94--IP-Adapter/h94--IP-Adapter/models/ip-adapter-faceid-portrait_sd15.bin"
    device = "cuda"

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
    
    image_paths = [os.path.join(args.input, name)for name in os.listdir(args.input)][:5] if os.path.isdir(args.input) \
        else [args.input]
    if args.embeds_path is None:
        app = FaceidAcquirer()
        faceid_embeds = app.get_multi_embeds(image_paths)
        print(faceid_embeds.shape)
    else:
        faceid_embeds = np.load(args.embeds_path)
        if len(faceid_embeds.shape) == 2 and len(faceid_embeds.shape) != 3:
            faceid_embeds = torch.from_numpy(faceid_embeds).unsqueeze(0)
        else:
            ValueError(f"faceid_embeds shape is error ==> {faceid_embeds.shape}")

    # load ip-adapter
    ip_model = IPAdapterFaceID(pipe, ip_ckpt, device, num_tokens=16, n_cond=len(image_paths))

    suffix = os.path.basename(image_paths[0]).split('.')[1]
    txt_path = image_paths[0].replace(suffix, 'txt')
    if os.path.exists(txt_path):
        with open(txt_path, 'r')as f:
            prompt = f.readlines()[0]
    prompt = prompt if args.prompt is None else args.prompt
    # negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

    print(f"prompt:{prompt}\n negative_prompt:{args.negative_prompt}")
    images = ip_model.generate(
        prompt=prompt, 
        # negative_prompt=negative_prompt, 
        faceid_embeds=faceid_embeds, 
        num_samples=4, 
        width=512, height=512, 
        num_inference_steps=30, 
        seed=2023
    )
    grid = image_grid(images, 2, 2)
    grid.save(args.save_path)
    print(f"result has saved in {args.save_path}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="image_path or image_dir for 1 portrait")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--embeds_path", type=str, default=None)
    args = parser.parse_args()
    if os.path.isdir(args.input):
        args.save_path = os.path.join(os.path.dirname(os.path.dirname(args.input))+'_output', 'faceid_portrait.jpg') if args.save_path is None else args.save_path
    else:
        args.save_path = os.path.join(os.path.dirname(args.input)+'_output', os.path.basename(args.input)) if args.save_path is None else args.save_path
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    main(args)
