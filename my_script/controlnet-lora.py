
import argparse
import cv2
from PIL import Image
import torch
from torchvision import transforms

from diffusers import DDIMScheduler, AutoencoderKL
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from my_script.util import image_grid


control_model_dict = {
    'hed': r'/mnt/nfs/file_server/public/mingjiahui/models/lllyasviel--sd-controlnet-hed/', 
}


def get_control_image(mode, image: Image.Image) -> Image.Image:
    if mode=='hed':
        from controlnet_aux import HEDdetector
        hed = HEDdetector.from_pretrained(r'/mnt/nfs/file_server/public/mingjiahui/models/lllyasviel--Annotators/')
        image = hed(image)      # input: PIL.Image; output: PIL.Image
    return image


def main(args):
    # init
    source_dir = r'/mnt/nfs/file_server/public/mingjiahui/models'
    base_model_path = fr'{source_dir}/Lykon--DreamShaper/'      # "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = fr"{source_dir}/stabilityai--sd-vae-ft-mse"
    device = "cuda"
    
    # load controlnet
    controlnet_model_path = control_model_dict[args.control_mode]
    print(f"\033[91m {controlnet_model_path} \033[0m")
    controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)

    # load sd
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
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    ).to(device)

    # load lora
    pipe.load_lora_weights(
        os.path.dirname(args.lora),
        weight_name=os.path.basename(args.lora), 
        adapter_name="style"
        )
    pipe.set_adapters(["style"], adapter_weights=[args.lora_weight])

    # generate image
    prompt = "closeup photo of a man wearing a white shirt in a garden, high quality, diffuse light, highly detailed, 4k"
    negative_prompt = "blurry, malformed, distorted, naked"

    transform = transforms.Compose([
        transforms.Resize(512)
    ])
    image = transform(Image.open(args.input))
    hed_map = get_control_image(args.control_mode, image)

    generator = torch.Generator(device).manual_seed(42)
    images = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        num_images_per_prompt=4, 
        num_inference_steps=30, 
        generator=generator, 
        guidance_scale=6,
        # width=512, height=512, 
        # control input
        image=hed_map,
        controlnet_conditioning_scale=args.control_scale
    ).images
    grid = image_grid(images, 2, 2)

    save_name = f"ControlNet-{args.control_mode}"
    save_name += f"-{os.path.basename(args.input).split('.')[0]}"
    save_name += f"-lora_{args.lora_weight}"
    save_name += f"-control_scale{args.control_scale}"
    save_path = os.path.join(args.output, save_name+'.jpg')
    grid.save(save_path)
    print(f"result has saved in {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="image path")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--control_mode", type=str, required=True)
    parser.add_argument("--control_scale", type=float, default=1.0)
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--lora_weight", type=float, default=1.0)
    args = parser.parse_args()
    args.output = os.path.dirname(args.input) + '_only_control_lora_output' \
        if args.output is None else args.output
    os.makedirs(args.output, exist_ok=True)
    print(args.output)
    main(args)
