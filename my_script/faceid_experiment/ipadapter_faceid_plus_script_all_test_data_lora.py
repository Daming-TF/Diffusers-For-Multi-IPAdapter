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
from my_script.util.util import FaceidAcquirer, image_grid
# from my_script.unetfix import CostomUNet2DConditionModel
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from ip_adapter.attention_processor_faceid import LoRAAttnProcessor

prompt_template = {
    '3dexaggeration_sd': "seekoo_gds,{},normcore,dynamic balance,punk,dotted,16k,3D,octane rendering,C4D,Zbrush,detailed character illustrations,animated gifs,in the style of graphic design-inspired illustrations,interactive experiences,best quality,focused,depth,incredibly detailed,masterpiece,soft light",
    '3dpixel2_sd': "seekoo_gds, Minimalism, close shot, LEGO, {}, masterpiece, highest quality, 16k",
    # 'joyful_cartoon': "seekoo_gds, 3d rending work, cartoon, pixar animation, 16k, masterpiece, best quality, sharp, focused, depth, incredibly detailed, soft light, {}" ,
    'graffitisplash_sd': "low saturation,morandi color,flat art,Graffiti style,paint,{},16k,masterpiece,best quality",
    'papercutout_sd': "seekoo_gds, paper illustration, flat illustration, 16k, masterpiece, best quality, sharp, {}",
}
negative_prompt_template = {
    '3dexaggeration_sd': '',
    '3dpixel2_sd': "BadDream,high saturation,low quality,low resolution,bad art,poor detailing,ugly,disfigured,text,watermark,signature,bad proportions,bad anatomy,duplicate,cropped, cut off, extra hands, extra arms, extra legs, poorly drawn face, unnatural pose, out of frame, unattractive, twisted body, extra limb, missing limb, mangled, malformed limbs",
    # 'joyful_cartoon': "BadDream",
    'graffitisplash_sd': "BadDream,high saturation,low quality,low resolution,bad art,poor detailing,ugly,disfigured,text,watermark,signature,bad proportions,bad anatomy,duplicate,cropped, cut off, extra hands, extra arms, extra legs, poorly drawn face, unnatural pose, out of frame, unattractive, twisted body, extra limb, missing limb, mangled, malformed limbs",
    'papercutout_sd': "BadDream, low quality, low resolution, bad art, poor detailing, ugly, disfigured, text, watermark, signature, bad proportions, bad anatomy, duplicate, cropped, cut off, extra hands, extra arms, extra legs, poorly drawn face, unnatural pose, out of frame, unattractive, twisted body, extra limb, missing limb, mangled, malformed limbs"
}

lora_dir = r"/mnt/nfs/file_server/public/guonan/lora10/11"
lora_dict = {
    '3dexaggeration_sd': rf"{lora_dir}/3D_Exaggeration_V2_lr5.0e-04_elast.safetensors",
    '3dpixel2_sd': rf"{lora_dir}/3D_Pixel_lr5.0e-04_e80.safetensors",
    # 'joyful_cartoon': rf"{lora_dir}/Joyful_Cartoon.safetensors",
    'graffitisplash_sd': rf"{lora_dir}/Graffiti_Splash_768_lr5.0e-04_e8.safetensors",
    'papercutout_sd': rf"{lora_dir}/Paper_Cutout.safetensors",
}

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(768)
])


def main(args):
    v2 = args.v2
    source_dir = r'/mnt/nfs/file_server/public/mingjiahui/models'
    # Lykon--DreamShaper/
    base_model_path = fr'{source_dir}/SG161222--Realistic_Vision_V4.0_noVAE'      # "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = fr"{source_dir}/stabilityai--sd-vae-ft-mse"
    image_encoder_path = fr"{source_dir}/laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
    ip_ckpt = fr"{source_dir}/h94--IP-Adapter/faceid/ip-adapter-faceid-plus_sd15.bin" if not v2 \
        else fr"{source_dir}/h94--IP-Adapter/faceid/ip-adapter-faceid-plusv2_sd15.bin"
    device = "cuda"
    app = FaceidAcquirer()
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

    # jiahui's modify
    image_dir = args.input
    print(image_dir)
    image_paths = [os.path.join(image_dir, name) for name in os.listdir(image_dir) \
                   if name.split('.')[1] in ['jpg', 'png', 'webp']]
    
    ori_lora = None
    for lora_id, lora_path in lora_dict.items():
        if ori_lora is not None:
            ip_model.pipe.delete_adapters(ori_lora)

        ip_model.pipe.load_lora_weights(
            os.path.dirname(lora_path),
            weight_name = os.path.basename(lora_path),
            adapter_name = lora_id
        )

        for image_path in image_paths:
            # (1)get reference image
            # faceid_embeds, face_image = get_face_embeds(transform(Image.open(image_path).convert("RGB")))
            faceid_embeds, face_image = app.get_face_embeds(Image.open(image_path).convert("RGB"))
            # (2)get prompt
            suffix = os.path.basename(image_path).split('.')[1]
            txt_path = image_path.replace(suffix, 'txt')
            with open(txt_path, 'r')as f:
                prompts = f.readlines()
                assert len(prompts)==1, "txt file looks happening some error"
            prompt = prompt_template[lora_id].format(prompts[0])
            
            # (3)processing
            result = None
            ip_model.pipe.set_adapters(lora_id, adapter_weights=0.0)
            image = ip_model.generate(
                prompt=prompt, 
                # negative_prompt=negative_prompt, 
                face_image=face_image, faceid_embeds=faceid_embeds, 
                shortcut=v2, s_scale=args.s_scale,
                num_samples=1, 
                width=768, height=768, 
                num_inference_steps=30, 
                seed=42, 
                guidance_scale=7.5,
            )[0]
            result = np.array(image)

            ip_model.pipe.set_adapters(lora_id, adapter_weights=0.8)
            image = ip_model.generate(
                prompt=prompt, 
                # negative_prompt=negative_prompt, 
                face_image=face_image, faceid_embeds=faceid_embeds, 
                shortcut=v2, s_scale=args.s_scale,
                num_samples=1, 
                width=768, height=768, 
                num_inference_steps=30, 
                seed=42, 
                guidance_scale=7.5,
            )[0]
            result = cv2.vconcat([result, np.array(image)])

            save_dir = os.path.join(args.output, lora_id)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, os.path.basename(image_path))
            Image.fromarray(result).save(save_path)
            print(f"result has saved in {save_path}")
        
        ori_lora = lora_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input image path")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--v2", action='store_true')
    parser.add_argument("--s_scale", type=float, default=1.0)
    # parser.add_argument("--lora_scale", type=f loat, default=0.8)
    args = parser.parse_args()
    args.output = args.input + '_lora_script_output' if args.output is None else args.output
    os.makedirs(args.output, exist_ok=True)
    print(f"V2:{args.v2}")
    print(f"result will saved in {args.output}")
    main(args)