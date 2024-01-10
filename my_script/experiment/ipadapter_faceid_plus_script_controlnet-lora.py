
import argparse
import cv2
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

from diffusers import DDIMScheduler, AutoencoderKL
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from my_script.util import get_face_embeds, image_grid
from my_script.ipadapter_faceid_plus import CostomIPAdapterFaceIDPlus
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus

control_model_dict = {
    'hed': r'/mnt/nfs/file_server/public/mingjiahui/models/lllyasviel--sd-controlnet-hed/', 
}

prompt_template = {
    '3d_pixel': "seekoo_gds, Minimalism, close shot, LEGO, {}, masterpiece, highest quality, 16k  <lora:3D_Pixel_lr5.0e-04_e80:0.8>",
    'joyful_cartoon': "seekoo_gds, 3d rending work, cartoon, pixar animation, 16k, masterpiece, best quality, sharp, focused, depth, incredibly detailed, soft light, {}" ,
    'paper_cutout': "seekoo_gds, paper illustration, flat illustration, 16k, masterpiece, best quality, sharp, {}",
}
negative_prompt_template = {
    '3d_pixel': "BadDream, high saturation,low quality, low resolution, bad art, poor detailing, ugly, disfigured, text, watermark, signature, bad proportions, bad anatomy, duplicate, cropped, cut off, extra hands, extra arms, extra legs, poorly drawn face, unnatural pose, out of frame, unattractive, twisted body, extra limb, missing limb, mangled, malformed limbs",
    'joyful_cartoon': "BadDream",
    'paper_cutout': "BadDream, low quality, low resolution, bad art, poor detailing, ugly, disfigured, text, watermark, signature, bad proportions, bad anatomy, duplicate, cropped, cut off, extra hands, extra arms, extra legs, poorly drawn face, unnatural pose, out of frame, unattractive, twisted body, extra limb, missing limb, mangled, malformed limbs"
}

lora_dict = {
    '3d_pixel': r"/mnt/nfs/file_server/public/mingjiahui/models/lora_1.5/3D_Pixel_lr5.0e-04_e80.safetensors",
    'joyful_cartoon': r"/mnt/nfs/file_server/public/mingjiahui/models/lora_1.5/Joyful_Cartoon.safetensors",
    'paper_cutout': r"/mnt/nfs/file_server/public/mingjiahui/models/lora_1.5/Paper_Cutout.safetensors",
}


def get_control_image(mode, image: Image.Image) -> Image.Image:
    if mode=='hed':
        from controlnet_aux import HEDdetector
        hed = HEDdetector.from_pretrained(r'/mnt/nfs/file_server/public/mingjiahui/models/lllyasviel--Annotators/')
        image = hed(image)      # input: PIL.Image; output: PIL.Image
    return image


def main(args):
    # init
    v2 = args.v2
    source_dir = r'/mnt/nfs/file_server/public/mingjiahui/models'
    base_model_path = fr'{source_dir}/Lykon--DreamShaper/'      # "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = fr"{source_dir}/stabilityai--sd-vae-ft-mse"
    image_encoder_path = fr"{source_dir}/laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
    ip_ckpt = fr"{source_dir}/h94--IP-Adapter/faceid/ip-adapter-faceid-plus_sd15.bin" if not v2 \
        else fr"{source_dir}/h94--IP-Adapter/faceid/ip-adapter-faceid-plusv2_sd15.bin"
    device = "cuda"
    print(f"\033[91m {ip_ckpt} \033[0m")

    # prapare Experiments param
    s, e, i = args.control_weights.split('-')
    control_weights = np.arange(float(s), float(e)+float(i), float(i))
    s, e, i = args.lora_weights.split('-')
    lora_weights = np.arange(float(s), float(e)+float(i), float(i))
    if os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        image_paths = [os.path.join(args.input, name) for name in os.listdir(args.input)]

    # load controlnet
    controlnet_model_path = control_model_dict[args.control_mode]
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
    )

    # load lora
    print(f"loadding lora....... ==> {args.lora}")
    if args.lora is not None:
        lora_path = lora_dict[args.lora]
        pipe.load_lora_weights(
            os.path.dirname(lora_path),
            weight_name=os.path.basename(lora_path), 
            adapter_name="style"
            )
        # pipe.set_adapters(["style"], adapter_weights=[args.lora_weight])

    # load ip-adapter
    ip_model = CostomIPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device)
    ip_model.set_lora_scale(args.faceid_lora_weight)

    # generate image
    prompt = args.prompt if args.lora is None else prompt_template[args.lora].format(args.prompt)
    negative_prompt = args.negative_prompt if args.lora is None else negative_prompt_template[args.lora]

    transform = transforms.Compose([
        transforms.Resize(512)
    ])

    for image_path in image_paths:
        image = transform(Image.open(image_path).convert("RGB"))
        hed_map = get_control_image(args.control_mode, image)
        hed_map.save(os.path.join(args.output, os.path.basename(image_path).split('.')[0]+'_hed.jpg'))
        faceid_embeds, face_image = get_face_embeds(image)

        result = None
        for lora_weight in lora_weights:
            pipe.set_adapters(["style"], adapter_weights=[lora_weight])
            h_concat = None
            for control_weight in control_weights:
                print(f"\n\prompt:{prompt}\nnegative_prompt:{negative_prompt}\nlora:{lora_weight}\tcontrol_weight:{control_weight}\n")
                image = ip_model.generate(
                    prompt=prompt, 
                    negative_prompt=negative_prompt, 
                    face_image=face_image, 
                    faceid_embeds=faceid_embeds, 
                    shortcut=v2, 
                    s_scale=args.s_scale,
                    num_samples=1, num_inference_steps=30, seed=42, guidance_scale=6,
                    # width=512, height=512, 
                    # control input
                    image=hed_map,
                    controlnet_conditioning_scale=control_weight
                )[0]

                h_concat = cv2.hconcat([h_concat, np.array(image)]) if h_concat is not None else np.array(image)
            result = cv2.vconcat([result, h_concat]) if result is not None else h_concat

        save_name = f"{os.path.basename(ip_ckpt).split('.')[0]}"
        save_name += f"-{os.path.basename(image_path).split('.')[0]}"
        save_name += f"-faceid_lora_{args.faceid_lora_weight}"
        save_name += f"-s_scale_{args.s_scale}"
        save_name += f"-lora_{args.lora}_{args.lora_weights}"
        save_name += f"-ControlNet_weight_{args.control_weights}"
        save_path = os.path.join(args.output, save_name+'.jpg')
        Image.fromarray(result).save(save_path)
        print(f"result has saved in {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="image path")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--v2", action='store_true')
    parser.add_argument("--s_scale", type=str, default=1.0)
    parser.add_argument("--faceid_lora_weight", type=float, default=1.0)
    parser.add_argument("--control_mode", type=str, required=True)
    parser.add_argument("--control_weights", type=str, default='0-0.5-0.1')
    parser.add_argument("--lora", type=str, default=None, help="Union['3d_pixel', 'joyful_cartoon', 'paper_cutout']")
    parser.add_argument("--lora_weights", type=str, default='0.5-1.0-0.1')
    parser.add_argument("--prompt", type=str, default="closeup photo of a persion wearing a white shirt in a garden, high quality, diffuse light, highly detailed, 4k")
    parser.add_argument("--negative_prompt", type=str, default="blurry, malformed, distorted, naked")
    args = parser.parse_args()
    args.output = os.path.dirname(args.input) + '_control_lora_output' \
        if args.output is None else args.output
    os.makedirs(args.output, exist_ok=True)
    print(args.output)
    main(args)
