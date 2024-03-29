import cv2
from PIL import Image
from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import argparse
import numpy as np
import cv2
import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from my_script.util.util import image_grid, FaceidAcquirer
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL
from ip_adapter.attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor
from ip_adapter.utils import is_torch2_available

USE_DAFAULT_ATTN = False # should be True for visualization_attnmap
if is_torch2_available() and (not USE_DAFAULT_ATTN):
    from ip_adapter.attention_processor_faceid import (
        LoRAAttnProcessor2_0 as LoRAAttnProcessor,
    )
    from ip_adapter.attention_processor_faceid import (
        LoRAIPAttnProcessor2_0 as LoRAIPAttnProcessor,
    )

class CostomIPAdapterFaceIDPlusXL(IPAdapterFaceIDPlusXL):
    def set_lora_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, LoRAAttnProcessor) or isinstance(attn_processor, LoRAIPAttnProcessor):
                attn_processor.lora_scale = scale


def main(args):
    # 1.init
    v2 = True
    source_dir = "/mnt/nfs/file_server/public/mingjiahui/models"
    base_model_path = "/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/"     # "SG161222/Realistic_Vision_V4.0_noVAE"
    # vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path =  f"{source_dir}/laion--CLIP-ViT-H-14-laion2B-s32B-b79K"      # "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    ip_ckpt = f"{source_dir}/h94--IP-Adapter/h94--IP-Adapter/sdxl_models/ip-adapter-faceid-plusv2_sdxl.bin"
    assert 'v2' in ip_ckpt, "sdxl only support plusv2, not supoort faceid plus"
    device = "cuda"

    app = FaceidAcquirer()
    s, e, i = args.faceid_lora_weight.split('-')
    faceid_lora_weights = np.arange(float(s), float(e)+float(i), float(i))
    s, e, i = args.ipscale.split('-')
    ip_scales = np.arange(float(s), float(e)+float(i), float(i))

    print(f" save path: {args.output}")
    print(f"faceid_lora_weights:{faceid_lora_weights}\tip_scales:{ip_scales}")
    print(f"\033[91m {ip_ckpt} \033[0m")

    # 2.load model
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

    # 3.load ip-adapter
    ip_model = CostomIPAdapterFaceIDPlusXL(pipe, image_encoder_path, ip_ckpt, device)

    # 4.prepare promot
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
    print(f"prompt:{prompt}")
    
    # prepare face id
    faceid_embeds, face_image = app.get_face_embeds(cv2.imread(args.input))

    # generate image
    result = None
    for ip_scale in ip_scales:
        hconcat = None
        for faceid_lora_weight in faceid_lora_weights:
            ip_model.set_lora_scale(faceid_lora_weight)
            image = ip_model.generate(
                prompt=prompt, negative_prompt=negative_prompt, face_image=face_image, faceid_embeds=faceid_embeds, 
                shortcut=v2, s_scale=1.0, scale=ip_scale,
                num_samples=1, width=1024, height=1024, num_inference_steps=30, seed=42,
            )[0]
            image = np.array(image)
            cv2.putText(
                image, 
                f"ip_{round(ip_scale, 2)}-lora_{round(faceid_lora_weight, 2)}", 
                (50,50), 
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0,0,255),
                thickness=3,
                )
            hconcat = cv2.hconcat([hconcat, image]) if hconcat is not None else image
        result = cv2.vconcat([result, hconcat]) if result is not None else hconcat

        Image.fromarray(result).save(args.output)
        print(f"result has saved in {args.output} ......")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input image path")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--faceid_lora_weight", type=str, default='0-1-0.2')
    parser.add_argument("--ipscale", type=str, default='0-1-0.2')
    args = parser.parse_args()
    assert os.path.isfile(args.input)
    args.output = os.path.join(
        os.path.dirname(args.input)+'__faceid_plusV2_xl_script', 
        f'lora_{args.faceid_lora_weight}-ipscale_{args.ipscale}', 
        os.path.basename(args.input),
        )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    main(args)
