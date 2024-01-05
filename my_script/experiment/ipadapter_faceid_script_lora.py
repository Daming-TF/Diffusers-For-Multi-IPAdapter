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
from ip_adapter.attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor


class CostomIPAdapterFaceID(IPAdapterFaceID):
    def set_lora_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, LoRAAttnProcessor) or isinstance(attn_processor, LoRAIPAttnProcessor):
                attn_processor.lora_scale = scale


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

    # jiahui'S modify       load lora    
    start, end, interval = args.lora_weights.split('-')
    lora_weights = np.arange(float(start), float(end)+float(interval), float(interval))
    # pipe.load_lora_weights(
    #     f"{source_dir}/h94--IP-Adapter/faceid", 
    #     weight_name="ip-adapter-faceid_sd15_lora.safetensors", 
    #     adapter_name="style"
    #     )
    # pipe.set_adapters(["style"], adapter_weights=[1.0])

    # load ip-adapter
    ip_model = CostomIPAdapterFaceID(pipe, ip_ckpt, device)

    # generate image
    prompt = "closeup photo of a man wearing a white shirt in a garden, high quality, diffuse light, highly detailed, 4k"
    negative_prompt = "blurry, malformed, distorted, naked"

    # jiahui's modify
    image_path = args.input
    faceid_embeds, _ = get_face_embeds(cv2.imread(image_path))

    result = None
    for lora_weight in lora_weights:
        # ip_model.pipe.set_adapters(["style"], adapter_weights=[lora_weight])
        ip_model.set_lora_scale(lora_weight)
        images = ip_model.generate(
            prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=4, width=512, height=512, num_inference_steps=30, seed=42, guidance_scale=6,
        )
        grid = np.array(image_grid(images, 2, 2))
        result = cv2.hconcat([result, grid]) if result is not None else grid

    save_name = f"{os.path.basename(ip_ckpt).split('.')[0]}"
    save_name += f"-{os.path.basename(args.input).split('.')[0]}"
    save_name += f"-lora_{args.lora_weights.replace('-','_')}"
    save_path = os.path.join(args.output, save_name+'.jpg')
    Image.fromarray(result).save(save_path)
    print(f"result has saved in {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input image path")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--lora_weights", type=str, default="0-1-0.2")
    args = parser.parse_args()
    args.output = os.path.dirname(args.input) + '_output' \
        if args.output is None else args.output
    os.makedirs(args.output, exist_ok=True)
    print(args.output)
    main(args)
    