import torch
from typing import List
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from PIL import Image
import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from ip_adapter.ip_adapter_faceid_separate import MLPProjModel
from safetensors import safe_open
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import AttnProcessor2_0 as AttnProcessor
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor
else:
    from  ip_adapter.attention_processor import AttnProcessor, IPAttnProcessor


class IPAdapterFaceID_TiToken:
    def __init__(self, 
            sd_pipe, device, 
            ip_ckpt, image_encoder_path,
            num_tokens=16, ti_num_tokens=4,
            n_cond=1, torch_dtype=torch.float16):
        self.device = device
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.ti_num_tokens = ti_num_tokens
        self.n_cond = n_cond
        self.torch_dtype = torch_dtype

        self.pipe = sd_pipe.to(self.device)
        self.set_ip_adapter()

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(self.device, dtype=self.torch_dtype)
        self.clip_image_processor = CLIPImageProcessor()
        self.init_proj()

        self.load_ip_adapter()

    def init_proj(self):
        self.image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            id_embeddings_dim=512,
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=self.torch_dtype)
        self.text_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            id_embeddings_dim=self.image_encoder.config.projection_dim,
            num_tokens=self.ti_num_tokens,
        ).to(self.device, dtype=self.torch_dtype)

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0, num_tokens=self.num_tokens*self.n_cond,
                ).to(self.device, dtype=self.torch_dtype)
        unet.set_attn_processor(attn_procs)

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.text_proj_model.load_state_dict(state_dict["text_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)

    @torch.inference_mode()
    def get_ip_image_embeds(self, faceid_embeds):
        multi_face = False
        if faceid_embeds.dim() == 3:
            multi_face = True
            b, n, c = faceid_embeds.shape
            faceid_embeds = faceid_embeds.reshape(b*n, c)

        faceid_embeds = faceid_embeds.to(self.device, dtype=self.torch_dtype)       # {1*5,512}
        ip_image_embeds = self.image_proj_model(faceid_embeds)      # {5,16,768}
        uncond_ip_image_embeds = self.image_proj_model(torch.zeros_like(faceid_embeds))
        if multi_face:
            c = ip_image_embeds.size(-1)
            ip_image_embeds = ip_image_embeds.reshape(b, -1, c)
            uncond_ip_image_embeds = uncond_ip_image_embeds.reshape(b, -1, c)
        
        return ip_image_embeds, uncond_ip_image_embeds
    
    @torch.inference_mode()
    def get_ip_text_embeds(self, face_image):
        clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=self.torch_dtype)
        clip_embeds = self.image_encoder(clip_image).image_embeds
        ip_text_embeds = self.text_proj_model(clip_embeds)
        uncond_clip_embeds = self.image_encoder(torch.zeros_like(clip_image)).image_embeds
        uncond_ip_text_embeds = self.text_proj_model(uncond_clip_embeds)
        return ip_text_embeds, uncond_ip_text_embeds


    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def generate(
        self,
        crop_images=None,
        faceid_embeds=None,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = faceid_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        ip_image_embeds, uncond_ip_image_embeds = self.get_ip_image_embeds(faceid_embeds)
        bs_embed, seq_len, _ = ip_image_embeds.shape
        ip_image_embeds = ip_image_embeds.repeat(1, num_samples, 1)
        ip_image_embeds = ip_image_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_ip_image_embeds = uncond_ip_image_embeds.repeat(1, num_samples, 1)
        uncond_ip_image_embeds = uncond_ip_image_embeds.view(bs_embed * num_samples, seq_len, -1)

        ip_text_embeds, uncond_ip_text_embeds = self.get_ip_text_embeds(crop_images)
        bs_embed, seq_len, _ = ip_text_embeds.shape
        ip_text_embeds = ip_text_embeds.repeat(1, num_samples, 1)
        ip_text_embeds = ip_text_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_ip_text_embeds = uncond_ip_text_embeds.repeat(1, num_samples, 1)
        uncond_ip_text_embeds = uncond_ip_text_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([ip_text_embeds, prompt_embeds_, ip_image_embeds], dim=1)
            negative_prompt_embeds = torch.cat([uncond_ip_text_embeds, negative_prompt_embeds_, uncond_ip_image_embeds], dim=1)

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images
