from diffusers import StableDiffusionXLPipeline
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms

import os
import sys
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from ip_adapter import IPAdapterPlusXL
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline
from ip_adapter.resampler import Resampler

from insightface.app import FaceAnalysis
from ip_adapter.utils import get_generator


class IPAdapterPlusXLCostom(IPAdapterPlusXL):
    def __init__(self, sd_pipe, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_ip_adapter()
        
    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=512,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, face_embeds):
        face_embeds = face_embeds.to(self.device, dtype=torch.float16)  
        image_prompt_embeds = self.image_proj_model(face_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(face_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    def generate(
        self,
        face_embeds,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(face_embeds)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


if __name__ == '__main__':
    source_dir = '/mnt/nfs/file_server/public/mingjiahui/models'
    base_model_path = f"/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0"
    ip_ckpt = f"{source_dir}/h94--IP-Adapter/h94--IP-Adapter/sdxl_models/ip-adapter-faceid-portrait_sdxl_unnorm.bin" # a experimental version
    device = "cuda"

    app = FaceAnalysis(name='/home/mingjiahui/.insightface/models/buffalo_l/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
    ])

    # load SDXL pipeline
    pipe = StableDiffusionXLCustomPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )

    # load ip-adapter
    ip_model = IPAdapterPlusXLCostom(pipe, ip_ckpt, device, num_tokens=16)

    image_paths = []
    image_dirs = [
        "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/famous",
        "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/average_id",
    ]
    for image_dir in image_dirs:
        image_paths += [os.path.join(image_dir, name) for name in os.listdir(image_dir) \
            if name.split('.')[-1] not in ['json', 'txt', 'npy']]

    for image_path in image_paths:
        image_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/average_id/guonan.jpg"
        txt_path = image_path.replace(os.path.basename(image_path).split('.')[-1], 'txt')
        with open(txt_path, 'r')as f:
            prompts = f.readlines()
            assert len(prompts)==1
            prompt = prompts[0]
        face_image = transform(Image.open(image_path).convert("RGB"))
        face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        if len(face_info) == 0:
            print(f"no face find ==> {image_path}")
            continue
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]   # only use the maximum face
        face_emb = torch.from_numpy(face_info.embedding).unsqueeze(0).unsqueeze(0)
        images = ip_model.generate(
            face_embeds=face_emb, 
            num_samples=1, 
            num_inference_steps=30, 
            seed=42,
            prompt=prompt
        )
        # grid = image_grid(images, 1, 2)
        # save_path = "/home/mingjiahui/project/IpAdapter_mjh/ip-adapter/data/debug/debug.jpg"
        save_dir = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/face_similarity_result/ip-adapter-faceid-portrait_sdxl_unnorm"
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        images[0].save(save_path)
        print(f"result has saved in {save_path}")
