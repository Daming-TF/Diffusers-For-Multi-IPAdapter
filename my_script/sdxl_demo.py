import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from my_script.util.util import image_grid

base_model_path = r"/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/"       # "SG161222/RealVisXL_V3.0"
ip_ckpt = r"/mnt/nfs/file_server/public/mingjiahui/models/h94--IP-Adapter/faceid/ip-adapter-faceid_sdxl.bin"
save_path = "./data/other/debug.jpg"
lora_path = "/mnt/nfs/file_server/public/mingjiahui/models/sdxl_lora_civitai/ptrn-no1_style_V1.0.safetensors"
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
).to(device)
pipe.load_lora_weights(
    os.path.dirname(lora_path),
    weight_name=os.path.basename(lora_path),
    adapter_name='style'
)
pipe.set_adapters(adapter_names='style', adapter_weights=1.0)

prompt = "closeup photo of a man wearing a white shirt in a garden, high quality, diffuse light, highly detailed, 4k"
negative_prompt = "blurry, malformed, distorted, naked" 

images = pipe(prompt, num_images_per_prompt=4, num_inference_steps=20).images
grid = image_grid(images)
grid.save(save_path)