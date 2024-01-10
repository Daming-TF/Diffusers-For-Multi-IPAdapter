
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import argparse

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from my_script.util import image_grid
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL, USE_DAFAULT_ATTN
from ip_adapter.attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor
from ip_adapter.utils import is_torch2_available


if is_torch2_available() and (not USE_DAFAULT_ATTN):
    from ip_adapter.attention_processor_faceid import (
        LoRAAttnProcessor2_0 as LoRAAttnProcessor,
    )
    from ip_adapter.attention_processor_faceid import (
        LoRAIPAttnProcessor2_0 as LoRAIPAttnProcessor,
    )
class CostomIPAdapterFaceIDXL(IPAdapterFaceIDXL):
    def set_lora_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, LoRAAttnProcessor) or isinstance(attn_processor, LoRAIPAttnProcessor):
                attn_processor.lora_scale = scale


def main(args):
    # 1. get face embeds
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    image = cv2.imread(args.input)        # r"./data/all_test_data/wangbaoqiang.jpg"
    faces = app.get(image)

    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

    # 2. load sdxl
    base_model_path = r"/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/"       # "SG161222/RealVisXL_V3.0"
    ip_ckpt = r"/mnt/nfs/file_server/public/mingjiahui/models/h94--IP-Adapter/faceid/ip-adapter-faceid_sdxl.bin"
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
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        add_watermarker=False,
    )

    # 3. load ip-adapter
    ip_model = CostomIPAdapterFaceIDXL(pipe, ip_ckpt, device)

    # 4. generate image
    prompt = "closeup photo of a man wearing a white shirt in a garden, high quality, diffuse light, highly detailed, 4k"
    negative_prompt = "blurry, malformed, distorted, naked"

    images = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=4,
        width=1024, height=1024,
        num_inference_steps=30, guidance_scale=7.5, seed=42
    )
    grid = image_grid(images, 2, 2)
    image_dir_name = os.path.basename(os.path.dirname(args.input))
    save_dir = os.path.join(args.output, image_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(args.input))
    grid.save(save_path)
    print(f"result has saved in ==> {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input image path")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    args.output = os.path.dirname(os.path.dirname(args.input)) + '_output' \
        if args.output is None else args.output
    os.makedirs(args.output, exist_ok=True)
    main(args)
