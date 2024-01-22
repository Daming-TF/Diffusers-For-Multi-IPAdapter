import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from my_script.util.util import get_face_embeds, image_grid
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID


def main():
    source_dir = r'/mnt/nfs/file_server/public/mingjiahui/models'
    base_model_path =f"{source_dir}/SG161222--Realistic_Vision_V4.0_noVAE/"   # "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = f"{source_dir}/stabilityai--sd-vae-ft-mse/"   # "stabilityai/sd-vae-ft-mse"
    ip_ckpt = f"{source_dir}/ip-adapter-faceid-portrait_sd15.bin"
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
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )


    # load ip-adapter
    ip_model = IPAdapterFaceID(pipe, ip_ckpt, device, num_tokens=16, n_cond=5)

    # generate image
    prompt = "photo of a woman in red dress in a garden"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

    images = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=4, width=512, height=512, num_inference_steps=30, seed=2023
    )
    grid = image_griid(images, (2,2))
    grid.save('')
    

if __name__ == '__main__':
    main()