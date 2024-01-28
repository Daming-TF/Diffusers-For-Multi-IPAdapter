import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from my_script.util.util import image_grid

base_model_path = r"/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/"       # "SG161222/RealVisXL_V3.0"
save_path = "./data/other/debug.jpg"

# lora_path = "/mnt/nfs/file_server/public/mingjiahui/models/lora-civit/Children_Illustration_SDXL.safetensors"
# trigger = "Children's Illustration Style,"

lora_path = "/mnt/nfs/file_server/public/mingjiahui/models/lora-civit/liujiyou-SDXL.safetensors"
trigger = "Chinese ink painting, traditional media, liujiyou"

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
print(F"loading lora...... ==> {os.path.basename(lora_path)}")
pipe.load_lora_weights(
    os.path.dirname(lora_path),
    weight_name=os.path.basename(lora_path),
    adapter_name='style'
)
pipe.set_adapters(adapter_names=['style'], adapter_weights=[1.0])

prompt = trigger + "1 dog"

<<<<<<< Updated upstream
images = pipe(prompt, num_images_per_prompt=4, num_inference_steps=30, height=1024, width=1024).images
grid = image_grid(images, 2, 2)
grid.save(save_path)
print(f"result has saved in {save_path}")
=======
images = pipe(prompt, num_images_per_prompt=4, num_inference_steps=20).images
grid = image_grid(images, 2, 2)
grid.save(save_path)
>>>>>>> Stashed changes
