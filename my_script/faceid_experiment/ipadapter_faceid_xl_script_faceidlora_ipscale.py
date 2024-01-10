import torch
from PIL import Image
import cv2
import numpy as np
import argparse
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from my_script.util import get_face_embeds, image_grid
from my_script.ipadapter_faceid_xl import CostomIPAdapterFaceIDXL


def main(args):
    base_model_path = r"/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/"       # "SG161222/RealVisXL_V3.0"
    ip_ckpt = r"/mnt/nfs/file_server/public/mingjiahui/models/h94--IP-Adapter/faceid/ip-adapter-faceid_sdxl.bin"
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

    print("loading sdxl......")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        add_watermarker=False,
    )

    # load ip-adapter
    ip_model = CostomIPAdapterFaceIDXL(pipe, ip_ckpt, device)

    # prepare data
    # 1.get face imbeds
    faceid_embeds, _ = get_face_embeds(cv2.imread(args.input))
    # 2.prompt
    suffix = os.path.basename(args.input).split('.')[1]
    txt_path = args.input.replace(suffix, 'txt')
    with open(txt_path, 'r')as f:
        txt_content = f.readlines()
    assert len(txt_content)==1
    prompt = txt_content[0]
    
    s, e, i = args.faceid_lora_weights.split('-')
    faceid_lora_weights = np.arange(float(s), float(e)+float(i), float(i))
    s, e, i = args.ipscales.split('-')
    ip_scales = np.arange(float(s), float(e)+float(i), float(i))

    result = None
    for ip_scale in ip_scales:
        hconcat = None
        for faceid_lora_weight in faceid_lora_weights:
            ip_model.set_lora_scale(faceid_lora_weight)
            image = ip_model.generate(
                prompt=prompt, faceid_embeds=faceid_embeds, num_samples=1, \
                width=1024, height=1024, \
                num_inference_steps=30, seed=42, guidance_scale=6,
                scale = ip_scale
            )[0]

            hconcat = cv2.hconcat([hconcat, np.array(image)]) if hconcat is not None else np.array(image)
        result = cv2.vconcat([result, hconcat]) if result is not None else hconcat
            
        image_dir_name = os.path.basename(os.path.dirname(args.input))
        save_dir = os.path.join(args.output, image_dir_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(args.input))
        Image.fromarray(result).save(save_path)
        print(f"result has saved in ==> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input image image for test data V2")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--faceid_lora_weights", type=str, default='0-1-0.2')
    parser.add_argument("--ipscales", type=str, default='0-1-0.2')
    args = parser.parse_args()
    args.output = os.path.join(os.path.dirname(os.path.dirname(args.input)) + '_param_explore', 'faceid_lora-ipscale') \
        if args.output is None else args.output
    os.makedirs(args.output, exist_ok=True)
    print(args.output)
    main(args)
    