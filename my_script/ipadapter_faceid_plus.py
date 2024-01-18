
import cv2
from PIL import Image
import torch
import argparse
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from my_script.util.util import FaceidAcquirer, image_grid
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus
from ip_adapter.attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor


class CostomIPAdapterFaceIDPlus(IPAdapterFaceIDPlus):
    def set_lora_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, LoRAAttnProcessor) or isinstance(attn_processor, LoRAIPAttnProcessor):
                attn_processor.lora_scale = scale


def main(args):
    v2 = args.v2
    source_dir = r'/mnt/nfs/file_server/public/mingjiahui/models'
    base_model_path = fr'{source_dir}/Lykon--DreamShaper/'      # "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = fr"{source_dir}/stabilityai--sd-vae-ft-mse"
    image_encoder_path = fr"{source_dir}/laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
    ip_ckpt = fr"{source_dir}/h94--IP-Adapter/faceid/ip-adapter-faceid-plus_sd15.bin" if not v2 \
        else fr"{source_dir}/h94--IP-Adapter/faceid/ip-adapter-faceid-plusv2_sd15.bin"
    device = "cuda"
    app = FaceidAcquirer()
    print(f"\033[91m {ip_ckpt} \033[0m")

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
    print(f"loadding lora....... ==> {args.lora}")
    if args.lora is not None:
        pipe.load_lora_weights(
            os.path.dirname(args.lora),
            weight_name=os.path.basename(args.lora), 
            adapter_name="style"
            )
        pipe.set_adapters(["style"], adapter_weights=[args.lora_weight])

    # load ip-adapter
    ip_model = CostomIPAdapterFaceIDPlus(pipe, image_encoder_path, ip_ckpt, device)
    ip_model.set_lora_scale(args.faceid_lora_weight)

    # generate image
    prompt = args.prompt
    negative_prompt = args.negative_prompt

    faceid_embeds, face_image = app.get_face_embeds(cv2.imread(args.input))

    images = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, face_image=face_image, faceid_embeds=faceid_embeds, shortcut=v2, s_scale=args.s_scale,
        num_samples=4, width=512, height=512, num_inference_steps=30, seed=42, guidance_scale=6,
    )
    grid = image_grid(images, 2, 2)

    save_name = f"{os.path.basename(ip_ckpt).split('.')[0]}"
    save_name += f"-{os.path.basename(args.input).split('.')[0]}"
    save_name += f"-lora_1"
    save_name += f"-s_scale_{args.s_scale}"
    save_path = os.path.join(args.output, save_name+'.jpg')
    save_path = os.path.join(args.output, f"{os.path.basename(ip_ckpt).split('.')[0]}-{os.path.basename(args.input)}")
    grid.save(save_path)
    print(f"result has saved in {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="image path")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--v2", action='store_true')
    parser.add_argument("--s_scale", type=str, default=1.0)
    parser.add_argument("--faceid_lora_weight", type=float, default=1.0)
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--lora_weight", type=float, default=1.0)
    parser.add_argument("--prompt", type=str, default="closeup photo of a man wearing a white shirt in a garden, high quality, diffuse light, highly detailed, 4k")
    parser.add_argument("--negative_prompt", type=str, default="blurry, malformed, distorted, naked")

    args = parser.parse_args()
    args.output = os.path.dirname(args.input) + '_output' \
        if args.output is None else args.output
    os.makedirs(args.output, exist_ok=True)
    print(args.output)
    main(args)
