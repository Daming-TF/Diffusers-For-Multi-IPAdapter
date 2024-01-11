# import numpy as np
# from tqdm import tqdm
from PIL import Image
import os
import math
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
from copy import deepcopy

import torch
from torch import fft
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor2_0

import os
import sys
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
from my_script.util.util import LayerStatus
from ip_adapter.attention_processor import IPAttnProcessor2_0
from ip_adapter import IPAdapter


BLOCKS_COUNT = 0
LAYER_NUM = None


def save_hidden_feature(np_array, cka_dir):
    save_path = os.path.join(cka_dir, f'layer_{BLOCKS_COUNT-1}.npy')
    np.save(save_path, np_array)


def load_model(
        base_model_path, 
        image_encoder_path, 
        ip_ckpt,
        vae_model_path=None, 
        controlnet_model_path=None, 
        lora_path=None,
        device='cuda', 
        unet_load=False,
        blip_load=False,
        img2img=False,
        token_num=16,
        multi_ip=False,
        ):
    global LAYER_NUM
    LAYER_NUM = 70 if 'xl' in base_model_path else 16

    # 1.init scheduler, SD-1.5 only works on DDIM, while SDXL works while on DPM++ and default sampler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    # 3.load my define unet
    # unet
    from .IPAdapter import UNet2DConditionModelV1 as UNet2DConditionModel
    if unet_load is True:
        print(f'loading unet...... ==> {base_model_path}')
        unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder='unet',
        ).to(dtype=torch.float16)

    # 4.load SD pipeline
    if img2img:
        # from diffusers import StableDiffusionXLImg2ImgPipeline
        from .IPAdapterXL import StableDiffusionXLImg2ImgPipelineV1 as StableDiffusionXLImg2ImgPipeline
        print(f'loading sdxl-img2img...... ==> {base_model_path}') 
        LAYER_NUM = 70
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                base_model_path,
                unet=unet,
                scheduler=noise_scheduler,
                torch_dtype=torch.float16,

            )
    elif controlnet_model_path is not None:
        if 'xl' in base_model_path:
            print('\033[91m Using controlnet..... \033[0m')
            print(f'loading controlnet...... ==> {controlnet_model_path}')
            from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
            from .IPAdapterXL import StableDiffusionXLControlNetPipelineV1 as StableDiffusionXLControlNetPipeline
            controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
            print(f'loading sd...... ==> {base_model_path}')
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                base_model_path,
                unet=unet,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                scheduler=noise_scheduler,
            )
        else:
            print('\033[91m Controlnet is only support for sdxl, if you want to use ipadapter layer-wise control + controlnet please switch to sdxl model \033[0m')
            exit(0)
    else:
        print(f'loading sd-txt2img...... ==> {base_model_path}')        
        if 'xl' in base_model_path:
            from .IPAdapterXL import StableDiffusionXLPipelineV1
            pipe = StableDiffusionXLPipelineV1.from_pretrained(
                base_model_path, 
                torch_dtype=torch.float16, 
                add_watermarker=False,
                unet=unet,
                )
        
        else:
            # 2.load vae
            if vae_model_path is not None:
                print(f'loading vae...... ==> {vae_model_path}')
                vae = AutoencoderKL.from_pretrained(
                    vae_model_path,
                    # use_safetensors=True,
                    low_cpu_mem_usage=False,
                    device_map=None,
                ).to(dtype=torch.float16)

            from .IPAdapter import StableDiffusionPipelineV1
            pipe = StableDiffusionPipelineV1.from_pretrained(
                base_model_path,
                use_safetensors=True,
                torch_dtype=torch.float16,
                scheduler=noise_scheduler,
                unet=unet,
                vae=vae,
                feature_extractor=None,
                safety_checker=None
            )
    
    
    # 5.load lora
    if lora_path is not None:
        print(f'loading the lora......  ==> {lora_path}')
        pipe.load_lora_weights(lora_path)

    # # 7. load BLIP
    # if blip_load:
    #     print(f"\033[91m loading BLIP...... \033[0m")
    #     from .Blip import BLIP
    #     blip = BLIP()
    # else:
    #     blip = None

    # 8.load ip-adapter
    print(f'loading ipadapter ..... ==> {ip_ckpt}')
    if multi_ip:
        assert isinstance(image_encoder_path, list) and isinstance(ip_ckpt, list) \
            and len(image_encoder_path)==len(ip_ckpt)
        units_parm = []
        for image_encoder_path_, ip_ckpt_ in zip(image_encoder_path, ip_ckpt):
            unit_parm = {
                'image_encoder_path': image_encoder_path_,
                'ip_ckpt': ip_ckpt_,
                'num_tokens': 16 if 'plus' in ip_ckpt_ else 4,
            }
            units_parm.append(unit_parm)

        from .multi_ipadapter import MultiIpadapter
        ip_model = MultiIpadapter(pipe, units_parm, device)

    else:
        assert isinstance(image_encoder_path, str) and isinstance(ip_ckpt, str)
        if 'xl' in base_model_path:
            from .IPAdapterXL import IPAdapterPlusXLV1, IPAdapterXL, IPAdapterXLV1
            if 'plus' in ip_ckpt:
                ip_model = IPAdapterPlusXLV1(pipe, image_encoder_path, ip_ckpt, device, num_tokens=token_num)
            else:
                ip_model = IPAdapterXLV1(pipe, image_encoder_path, ip_ckpt, device)

        else:
            if 'ip-adapter_sd15' in os.path.basename(image_encoder_path):
                ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device, num_tokens=4)
            else:
                from .IPAdapter import IPAdapterV1
                ip_model = IPAdapterV1(pipe, image_encoder_path, ip_ckpt, device, 
                                        num_tokens=16)

    return ip_model


def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda() 

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask
    

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    return x_filtered.to(x.dtype)


# #  the below functions is from https://github.com/tencent-ailab/IP-Adapter/tree/main/ip_adapter
# class ImageProjModel(torch.nn.Module):
#     """Projection Model"""
#     def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
#         super().__init__()
#         self.cross_attention_dim = cross_attention_dim
#         self.clip_extra_context_tokens = clip_extra_context_tokens
#         self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
#         self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
#     def forward(self, image_embeds):
#         embeds = image_embeds
#         clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
#         clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
#         return clip_extra_context_tokens


# # FFN
# def FeedForward(dim, mult=4):
#     inner_dim = int(dim * mult)
#     return nn.Sequential(
#         nn.LayerNorm(dim),
#         nn.Linear(dim, inner_dim, bias=False),
#         nn.GELU(),
#         nn.Linear(inner_dim, dim, bias=False),
#     )
    
    
# def reshape_tensor(x, heads):
#     bs, length, width = x.shape
#     #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
#     x = x.view(bs, length, heads, -1)
#     # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
#     x = x.transpose(1, 2)
#     # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
#     x = x.reshape(bs, heads, length, -1)
#     return x


# class PerceiverAttention(nn.Module):
#     def __init__(self, *, dim, dim_head=64, heads=8):
#         super().__init__()
#         self.scale = dim_head**-0.5
#         self.dim_head = dim_head
#         self.heads = heads
#         inner_dim = dim_head * heads

#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)

#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
#         self.to_out = nn.Linear(inner_dim, dim, bias=False)


#     def forward(self, x, latents):
#         """
#         Args:
#             x (torch.Tensor): image features
#                 shape (b, n1, D)
#             latent (torch.Tensor): latent features
#                 shape (b, n2, D)
#         """
#         x = self.norm1(x)               # {1,257,768}
#         latents = self.norm2(latents)   # {1,16,768}
        
#         b, l, _ = latents.shape

#         q = self.to_q(latents)                          # {1,16,768}
#         kv_input = torch.cat((x, latents), dim=-2)      # {1,273,768}
#         k, v = self.to_kv(kv_input).chunk(2, dim=-1)    # {1,273,768}
        
#         q = reshape_tensor(q, self.heads)
#         k = reshape_tensor(k, self.heads)
#         v = reshape_tensor(v, self.heads)

#         # attention
#         scale = 1 / math.sqrt(math.sqrt(self.dim_head))
#         weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
#         weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
#         out = weight @ v
        
#         out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

#         return self.to_out(out)
    

# class Resampler(nn.Module):
#     def __init__(
#         self,
#         dim=1024,               # 768
#         depth=8,                # 4
#         dim_head=64,            # 64
#         heads=16,               # 12
#         num_queries=8,          # 16
#         embedding_dim=768,      # 1280
#         output_dim=1024,        # 768
#         ff_mult=4,              # 4
#     ):
#         super().__init__()

#         self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        
#         self.proj_in = nn.Linear(embedding_dim, dim)

#         self.proj_out = nn.Linear(dim, output_dim)
#         self.norm_out = nn.LayerNorm(output_dim)
        
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(
#                 nn.ModuleList(
#                     [
#                         PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
#                         FeedForward(dim=dim, mult=ff_mult),
#                     ]
#                 )
#             )

#     def forward(self, x):
#         latents = self.latents.repeat(x.size(0), 1, 1)

#         x = self.proj_in(x)
        
#         for attn, ff in self.layers:
#             latents = attn(x, latents) + latents
#             latents = ff(latents) + latents
            
#         latents = self.proj_out(latents)
#         return self.norm_out(latents)


# TODO BY JIAHUI:V2
#   [support multi ipadpater] 
class MultiIPAttnProcessor2_0(torch.nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, units_num=1, scale=1):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.ip_scale = scale
        self.ip_units_enable = None
        # midjourney
        self.ip_kv_norm = False

        #  nn.ModuleList
        self.to_k_ip = [nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False) for _ in range(units_num)]
        self.to_v_ip = [nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False) for _ in range(units_num)]

    def __call__(
        self,
        attn,
        hidden_states,                  # 2, 4096, 320   
        attention_mask=None,
        temb=None,
        # multi-ip param:
        encoder_hidden_states: Optional[List[List[torch.tensor]]] = None,     # List[2, 93, 768]   {8, 16, 2018}
        units_num_token: Optional[List[int]] = None,  
        weights_enable: Optional[List[int]] = None,     # layer-wise control
        cn_weights: Optional[List[float]] = None,     # fooocus
        denoise_control_enable: Optional[List[bool]] = None,
        layer_status: Optional[LayerStatus] = None ,
        # other param
        **kwargs
    ):  
        global BLOCKS_COUNT
        from ui_v2 import LAYER_NUM
        LAYER_NUM = LAYER_NUM
        BLOCKS_COUNT = 1 if BLOCKS_COUNT == LAYER_NUM else BLOCKS_COUNT + 1     # SDXL：70      SD15:16

        if not isinstance(encoder_hidden_states, list):
            print("Error: var 'encoder_hidden_states' is not list - maybe the code vision is old")
            exit(0)

        residual = hidden_states
        lora_scale=kwargs.pop('scale', 1)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states[0] is None else encoder_hidden_states[0][0].shape
        )       # 2, 93, 768

        # if attention_mask is not None:
        #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #     # scaled_dot_product_attention expects attention_mask shape to be
        #     # (batch, heads, source_length, target_length)
        #     attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # query = attn.to_q(hidden_states, scale=lora_scale)     # JIAHUI'S MODIFY # 2, 4096, 320
        query = attn.to_q(hidden_states)

        # JIAHUI'S MODIFY
        new_encoder_hidden_states = None
        if encoder_hidden_states is None:       # for denoising control
            encoder_hidden_states = [[hidden_states]]
        else:
            # get encoder_hidden_states, ip_hidden_states
            assert len(units_num_token)==len(encoder_hidden_states)==len(denoise_control_enable)
            ip_units_hidden_states = []
            for unit_num_token, unit_hidden_states, enable in zip(units_num_token, encoder_hidden_states, denoise_control_enable):
                ip_unit_hidden_states = []
                for encoder_hidden_states_ in unit_hidden_states:
                    end_pos = encoder_hidden_states_.shape[1] - unit_num_token
                    encoder_hidden_states_, ip_hidden_states_ = encoder_hidden_states_[:, :end_pos, :], encoder_hidden_states_[:, end_pos:, :]  # debug: 1).en..:   max:854.5     min:-809      2).ip: max:7.3594   min:-8.4844
                    if attn.norm_cross:
                        encoder_hidden_states_ = attn.norm_encoder_hidden_states(encoder_hidden_states_)    
                    
                    if new_encoder_hidden_states is not None:
                        assert torch.allclose(new_encoder_hidden_states, encoder_hidden_states_), 'Error: once inference just only 1 prompt'
                    
                    new_encoder_hidden_states = encoder_hidden_states_
                    ip_unit_hidden_states.append(ip_hidden_states_ if enable else None) 
                ip_units_hidden_states.append(ip_unit_hidden_states)
        # +++++++++++++++++++++++++++++
        key = attn.to_k(new_encoder_hidden_states)
        value = attn.to_v(new_encoder_hidden_states)
        # key = attn.to_k(new_encoder_hidden_states, lora_scale)      # 2,77,320
        # value = attn.to_v(new_encoder_hidden_states, lora_scale)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)






        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        # JIAHUI'S MODIFY
        # for ip-adapter
        assert len(ip_units_hidden_states) == len(self.ip_scale) == len(weights_enable) == len(cn_weights) \
            == self.ip_units_enable.count(True)
        unit_id = -1
        for unit_hidden_states, ip_scale, weight_enable , cn_weights_ in zip(ip_units_hidden_states, self.ip_scale, weights_enable, cn_weights):
            assert len(unit_hidden_states)==len(ip_scale)
            try:
                unit_id = self.ip_units_enable.index(True, unit_id+1)                                       # debug: unit_id:0
                to_k_ip = self.to_k_ip[unit_id].to(hidden_states.device, hidden_states.dtype)               # debug:    weight max:0.2549   min:-0.3096
                to_v_ip = self.to_v_ip[unit_id].to(hidden_states.device, hidden_states.dtype)
            except ValueError as e:     # ValueError is raised if True cannot be found
                print(f"{e}: unit enable seem to something error")
                exit(1)

            for ip_hidden_states, ip_scale_, weight_enable_ in zip(unit_hidden_states, ip_scale, weight_enable): 
                if ip_hidden_states is None:
                    continue
                ip_key = to_k_ip(ip_hidden_states)    
                ip_value = to_v_ip(ip_hidden_states)        # {2,16,640}
                
                ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # reference fooocus：
                # https://github.com/lllyasviel/Fooocus/blob/main/fooocus_extras/ip_adapter.py#L232
                # https://github.com/lllyasviel/Fooocus/blob/main/modules/flags.py#L32
                # cn_ip: (0.5, 0.6), cn_ip_face: (0.9, 0.75), cn_canny: (0.5, 1.0), cn_cpds: (0.5, 1.0)
                if self.ip_kv_norm:
                    ip_value_mean = torch.mean(ip_value, dim=1, keepdim=True)
                    ip_value_offset = ip_value - ip_value_mean

                    b, h, s, h_dim = ip_key.shape      # {2,10,16,64}  {2,16,640}
                    channel = h * h_dim     # h_dim
                    channel_penalty = float(channel) / 1280.0       # 0.5
                    weight = cn_weights_ * channel_penalty      # 0.3

                    ip_key = ip_key * weight
                    ip_value = ip_value_offset + ip_value_mean * weight     
                # ++++++++++++++++++++++++++++

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                ip_hidden_states = F.scaled_dot_product_attention(
                    query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )
                
                ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                ip_hidden_states = ip_hidden_states.to(query.dtype)

                # for statistics diff between text embeds and image embeds
                if layer_status is not None:
                    layer_status('hidden_states', hidden_states.min(),hidden_states.max(), \
                                hidden_states.mean(),hidden_states.var())
                    layer_status('ip_hidden_states', ip_hidden_states.min(),ip_hidden_states.max(), \
                                ip_hidden_states.mean(),ip_hidden_states.var())
                
                # hidden_states:
                # max: 1.6348   min: -1.0166    mean:0.0004
                # ip_hidden_states:
                # max: -0.0208  min: -6.1836    mean:-0.0208
                hidden_states +=  ip_hidden_states * ip_scale_ * weight_enable_[BLOCKS_COUNT-1]       # ip_hidden_states.mean() = -0.00208 hidden_states.mean（） = -0.0004
        
        # ++++++++++++++++++++++++++++++++++

        # linear proj
        # hidden_states = attn.to_out[0](hidden_states, scale=lora_scale)     # JIAHUI'S MODIFY
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    

# class IPAttnProcessor2_0(torch.nn.Module):
#     r"""
#     Attention processor for IP-Adapater for PyTorch 2.0.
#     Args:
#         hidden_size (`int`):
#             The hidden size of the attention layer.
#         cross_attention_dim (`int`):
#             The number of channels in the `encoder_hidden_states`.
#         scale (`float`, defaults to 1.0):
#             the weight scale of image prompt.
#         num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
#             The context length of the image features.
#     """

#     def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4):
#         super().__init__()

#         if not hasattr(F, "scaled_dot_product_attention"):
#             raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

#         self.hidden_size = hidden_size
#         self.cross_attention_dim = cross_attention_dim
#         self.scale = scale
#         self.num_tokens = num_tokens

#         self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
#         self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

#     def __call__(
#         self,
#         attn,
#         hidden_states,                  # 2, 4096, 320
#         encoder_hidden_states=None,     # 2, 93, 768
#         attention_mask=None,
#         temb=None,
#     ):
#         residual = hidden_states

#         if attn.spatial_norm is not None:
#             hidden_states = attn.spatial_norm(hidden_states, temb)

#         input_ndim = hidden_states.ndim

#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

#         batch_size, sequence_length, _ = (
#             hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
#         )       # 2, 93, 768

#         if attention_mask is not None:
#             attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
#             # scaled_dot_product_attention expects attention_mask shape to be
#             # (batch, heads, source_length, target_length)
#             attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

#         if attn.group_norm is not None:
#             hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         query = attn.to_q(hidden_states)        # 2, 4096, 320


#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         else:
#             # get encoder_hidden_states, ip_hidden_states
#             end_pos = encoder_hidden_states.shape[1] - self.num_tokens
#             encoder_hidden_states, ip_hidden_states = encoder_hidden_states[:, :end_pos, :], encoder_hidden_states[:, end_pos:, :]
#             if attn.norm_cross:
#                 encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

#         key = attn.to_k(encoder_hidden_states)      # 2,77,320
#         value = attn.to_v(encoder_hidden_states)

#         inner_dim = key.shape[-1]
#         head_dim = inner_dim // attn.heads

#         query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         # the output of sdp = (batch, num_heads, seq_len, head_dim)
#         # TODO: add support for attn.scale when we move to Torch 2.1
#         hidden_states = F.scaled_dot_product_attention(
#             query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#         )

#         hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#         hidden_states = hidden_states.to(query.dtype)
        
#         # for ip-adapter
#         ip_key = self.to_k_ip(ip_hidden_states)
#         ip_value = self.to_v_ip(ip_hidden_states)
        
#         ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         # the output of sdp = (batch, num_heads, seq_len, head_dim)
#         # TODO: add support for attn.scale when we move to Torch 2.1
#         ip_hidden_states = F.scaled_dot_product_attention(
#             query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
#         )
        
#         ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#         ip_hidden_states = ip_hidden_states.to(query.dtype)

#         hidden_states = hidden_states + self.scale * ip_hidden_states

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)

#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#         if attn.residual_connection:
#             hidden_states = hidden_states + residual

#         hidden_states = hidden_states / attn.rescale_output_factor

#         return hidden_states


# TODO BY JIAHUI: 2_3 suport "encoder_hidden_states" parameter input type is list
#   exp - [Single image rotation transform feature averaging]
#   exp - [multi image infernece]
#   exp - [fine control - add BLOCKS_COUNT]
class IPAttnProcessor2_1(IPAttnProcessor2_0):
    def __call__(
        self,
        attn,
        hidden_states,                  # 2, 4096, 320
        encoder_hidden_states=None,     # 2, 93, 768
        attention_mask=None,
        temb=None,
        mode='txt_img',    # choise in['txt_img', 'txt']
        weights_enable=None,
        cka_dir=None,
    ):  
        
        global BLOCKS_COUNT
        global LAYER_NUM
        if LAYER_NUM is None:
            # from ui import LAYER_NUM
            from ui_v2 import LAYER_NUM
            LAYER_NUM = LAYER_NUM

        BLOCKS_COUNT = 1 if BLOCKS_COUNT == LAYER_NUM else BLOCKS_COUNT + 1     # SDXL：70      SD15:16
        # print(f'BLOCKS_COUNT:{BLOCKS_COUNT}')
        # assert isinstance(encoder_hidden_states, list)

        if not isinstance(encoder_hidden_states, list):
            encoder_hidden_states = [encoder_hidden_states]
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states[0].shape
        )       # 2, 93, 768

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)        # 2, 4096, 320

        new_encoder_hidden_states = None
        ip_hidden_states_group = []
        # JIAHUI'S MODIFICATION: ADDING FOR LOOP FOR LIST OF REFERENCE IMAGES
        for encoder_hidden_states_ in encoder_hidden_states:
            if encoder_hidden_states_ is None:
                # encoder_hidden_states_ = hidden_states
                print('some error is happenced')
                exit(0)
            else:
                if mode == 'txt_img':
                    # get encoder_hidden_states, ip_hidden_states
                    end_pos = encoder_hidden_states_.shape[1] - self.num_tokens
                    encoder_hidden_states_, ip_hidden_states = encoder_hidden_states_[:, :end_pos, :], encoder_hidden_states_[:, end_pos:, :]
                    ip_hidden_states_group.append(ip_hidden_states)
                elif mode == 'txt':
                    pass
                else:
                    print("mode must choise in ['txt_img', 'img']")
                    exit(0)

                if attn.norm_cross:
                    encoder_hidden_states_ = attn.norm_encoder_hidden_states(encoder_hidden_states_)
            
                new_encoder_hidden_states = encoder_hidden_states_
        ########################################################### 

        key = attn.to_k(new_encoder_hidden_states)      # 2,77,320
        value = attn.to_v(new_encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # JIAHUI'S MODIFICATION: ADDING MARK's to insert the ip_adapter conditionally
        if mode == 'txt_img':
            # check
            if not isinstance(self.scale, list):
                self.scale = [self.scale]
            weights_enable = [[1]*LAYER_NUM] if weights_enable is None else weights_enable
            weights_enable = weights_enable if isinstance(weights_enable[0], list) else [weights_enable]
            assert len(self.scale)==len(ip_hidden_states_group)==len(weights_enable)

            total_scale = 0
            for ip_hidden_states, ip_scale, weights_enable_ in zip(ip_hidden_states_group, self.scale, weights_enable):
                assert len(weights_enable_)==LAYER_NUM
                # for ip-adapter
                ip_key = self.to_k_ip(ip_hidden_states)
                ip_value = self.to_v_ip(ip_hidden_states)

                ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                ip_hidden_states = F.scaled_dot_product_attention(
                    query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )

                ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                ip_hidden_states = ip_hidden_states.to(query.dtype)

                hidden_states = hidden_states + ip_scale * ip_hidden_states * weights_enable_[BLOCKS_COUNT-1]
                total_scale += ip_scale
            
            if total_scale > 2:
                print("'scale' setting is something error")
        #####################################################

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if cka_dir is not None:
            save_hidden_feature(deepcopy(hidden_states).cpu().numpy(), cka_dir)
            # if hidden_states.shape[0] == 2: 
            #     save_hidden_feature(deepcopy(hidden_states[1]).cpu().numpy(), cka_dir)
            # else:
            #     print('some error!!')
            #     exit(1)

        return hidden_states


# class AttnProcessor2_0(torch.nn.Module):
#     r"""
#     Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
#     """
#     def __init__(
#         self,
#         hidden_size=None,
#         cross_attention_dim=None,
#     ):
#         super().__init__()
#         if not hasattr(F, "scaled_dot_product_attention"):
#             raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

#     def __call__(
#         self,
#         attn,
#         hidden_states,
#         encoder_hidden_states=None,
#         attention_mask=None,
#         temb=None,
#         **kwargs,
#     ):
#         residual = hidden_states
#         lora_scale = kwargs.pop('scale', 1)

#         if attn.spatial_norm is not None:
#             hidden_states = attn.spatial_norm(hidden_states, temb)

#         input_ndim = hidden_states.ndim

#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

#         batch_size, sequence_length, _ = (
#             hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
#         )

#         if attention_mask is not None:
#             attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
#             # scaled_dot_product_attention expects attention_mask shape to be
#             # (batch, heads, source_length, target_length)
#             attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

#         if attn.group_norm is not None:
#             hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         query = attn.to_q(hidden_states, scale=lora_scale)      # JIAHUI'S MODIFY

#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         elif attn.norm_cross:
#             encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

#         key = attn.to_k(encoder_hidden_states, scale=lora_scale)        # JIAHUI'S MODIFY
#         value = attn.to_v(encoder_hidden_states, scale=lora_scale)      # JIAHUI'S MODIFY

#         inner_dim = key.shape[-1]
#         head_dim = inner_dim // attn.heads

#         query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         # the output of sdp = (batch, num_heads, seq_len, head_dim)
#         # TODO: add support for attn.scale when we move to Torch 2.1
#         hidden_states = F.scaled_dot_product_attention(
#             query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#         )

#         hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#         hidden_states = hidden_states.to(query.dtype)

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states, scale=lora_scale)     # JIAHUI'S MODIFY
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)

#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#         if attn.residual_connection:
#             hidden_states = hidden_states + residual

#         hidden_states = hidden_states / attn.rescale_output_factor

#         return hidden_states


# class CNAttnProcessor2_0:
#     r"""
#     Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
#     """

#     def __init__(self,  num_tokens=4):
#         if not hasattr(F, "scaled_dot_product_attention"):
#             raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
#         self.num_tokens = num_tokens

#     def __call__(
#         self,
#         attn,
#         hidden_states,
#         encoder_hidden_states=None,
#         attention_mask=None,
#         temb=None,
#         **kwargs,
#     ):
#         residual = hidden_states

#         if attn.spatial_norm is not None:
#             hidden_states = attn.spatial_norm(hidden_states, temb)

#         input_ndim = hidden_states.ndim

#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

#         batch_size, sequence_length, _ = (
#             hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
#         )

#         if attention_mask is not None:
#             attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
#             # scaled_dot_product_attention expects attention_mask shape to be
#             # (batch, heads, source_length, target_length)
#             attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

#         if attn.group_norm is not None:
#             hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         query = attn.to_q(hidden_states)

#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         else:
#             end_pos = encoder_hidden_states.shape[1] - self.num_tokens
#             encoder_hidden_states = encoder_hidden_states[:, :end_pos] # only use text
#             if attn.norm_cross:
#                 encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

#         key = attn.to_k(encoder_hidden_states)
#         value = attn.to_v(encoder_hidden_states)

#         inner_dim = key.shape[-1]
#         head_dim = inner_dim // attn.heads

#         query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
#         value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

#         # the output of sdp = (batch, num_heads, seq_len, head_dim)
#         # TODO: add support for attn.scale when we move to Torch 2.1
#         hidden_states = F.scaled_dot_product_attention(
#             query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
#         )

#         hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
#         hidden_states = hidden_states.to(query.dtype)

#         # linear proj
#         hidden_states = attn.to_out[0](hidden_states)
#         # dropout
#         hidden_states = attn.to_out[1](hidden_states)

#         if input_ndim == 4:
#             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#         if attn.residual_connection:
#             hidden_states = hidden_states + residual

#         hidden_states = hidden_states / attn.rescale_output_factor

#         return hidden_states


# TODO: BY JIAHUI:V2
#   [support multi ipadpater + controlnet] 
class CNAttnProcessor2_1:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self,):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states,
        # encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        # JIAHUI'S MODIFY
        encoder_hidden_states: Optional[List[List[torch.tensor]]] = None,
        units_num_token: Optional[List[int]] = None,   
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states[0][0].shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        # JIAHUI'S MODIFY
        # The prompt is the same for multiple inputs from different units, so it can be taken only once
        if encoder_hidden_states is None:
            print('This brench is only support multi-ipadapter,so encoder_hidden_states will not None')
        else:
            assert len(encoder_hidden_states) == len(units_num_token)
            for unit_hidden_states, num_tokens in zip(encoder_hidden_states, units_num_token):
                end_pos = unit_hidden_states[0].shape[1] - num_tokens
                encoder_hidden_states = unit_hidden_states[0][:, :end_pos] # only use text
                if attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
                break
        # ++++++++++++++++++++++++++

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# class IPAdapter:
#     def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4, blip=None):
#         # Qirui's modified: merge single images and multiple images
#         self.IPAttnProcessor = IPAttnProcessor2_1

#         self.device = device
#         self.image_encoder_path = image_encoder_path
#         self.ip_ckpt = ip_ckpt
#         self.num_tokens = num_tokens

#         self.pipe = sd_pipe.to(self.device)
#         self.blip = blip
#         self.set_ip_adapter()

#         # load image encoder
#         print(f'loading vit...... ==> {self.image_encoder_path}')
#         self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(self.device,
#                                                                                                        dtype=torch.float16)
#         self.clip_image_processor = CLIPImageProcessor()
#         # image proj model
#         self.image_proj_model = self.init_proj()

#         self.load_ip_adapter()

#     def init_proj(self):
#         image_proj_model = ImageProjModel(
#             cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
#             clip_embeddings_dim=self.image_encoder.config.projection_dim,
#             clip_extra_context_tokens=self.num_tokens,
#         ).to(self.device, dtype=torch.float16)
#         return image_proj_model

#     def set_ip_adapter(self):
#         unet = self.pipe.unet
#         attn_procs = {}
#         for name in unet.attn_processors.keys():
#             cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
#             if name.startswith("mid_block"):
#                 hidden_size = unet.config.block_out_channels[-1]
#             elif name.startswith("up_blocks"):
#                 block_id = int(name[len("up_blocks.")])
#                 hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
#             elif name.startswith("down_blocks"):
#                 block_id = int(name[len("down_blocks.")])
#                 hidden_size = unet.config.block_out_channels[block_id]
#             if cross_attention_dim is None:
#                 attn_procs[name] = AttnProcessor2_0()
#             else:
#                 attn_procs[name] = self.IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
#                                                    scale=1.0, num_tokens=self.num_tokens).to(self.device,
#                                                                                              dtype=torch.float16)
#         unet.set_attn_processor(attn_procs)
#         if hasattr(self.pipe, "controlnet"):
#             self.pipe.controlnet.set_attn_processor(CNAttnProcessor2_0(num_tokens=self.num_tokens))

#     def load_ip_adapter(self):
#         state_dict = torch.load(self.ip_ckpt, map_location="cpu")
#         self.image_proj_model.load_state_dict(state_dict["image_proj"])
#         ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
#         ip_layers.load_state_dict(state_dict["ip_adapter"])

#         # transfer to iplora format
#         from safetensors.torch import save_file
#         state = {}
#         for ori_key, cur_value in zip(
#             self.pipe.unet.attn_processors.keys(), ip_layers.state_dict().values()):
#             state[ori_key] = cur_value
#         new_state = dict(sorted(state.items()))
#         iplora_sd = {}
#         iplora_sd.setdefault("image_proj", state_dict["image_proj"])
#         iplora_sd.setdefault("ip_adapter", {})
#         for new_key, new_value in zip(ip_layers.state_dict().keys(), new_state.values()):
#             iplora_sd['ip_adapter'][new_key] = new_value

#         save_path = os.path.join(os.path.dirname(self.ip_ckpt), 'iplora-'+os.path.basename(self.ip_ckpt))
#         torch.save(iplora_sd, save_path)
#         print(f'saveing ipadapter model for iplora to ==> {save_path}')


#     @torch.inference_mode()
#     def get_image_embeds(self, pil_image):
#         if isinstance(pil_image, Image.Image):
#             pil_image = [pil_image]
#         clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
#         clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
#         image_prompt_embeds = self.image_proj_model(clip_image_embeds)
#         uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
#         return image_prompt_embeds, uncond_image_prompt_embeds

#     def set_scale(self, scale):
#         for attn_processor in self.pipe.unet.attn_processors.values():
#             if isinstance(attn_processor, self.IPAttnProcessor):
#                 attn_processor.scale = scale

#     def generate(
#             self,
#             pil_image,
#             prompt=None,
#             negative_prompt=None,
#             scale=1.0,
#             num_samples=4,
#             seed=-1,
#             guidance_scale=7.5,
#             num_inference_steps=30,
#             **kwargs,
#     ):
#         self.set_scale(scale)
#         if isinstance(pil_image, Image.Image):
#             num_prompts = 1
#         else:
#             num_prompts = len(pil_image)

#         if prompt is None:
#             prompt = "best quality, high quality"
#         if negative_prompt is None:
#             negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

#         if not isinstance(prompt, List):
#             prompt = [prompt] * num_prompts
#         if not isinstance(negative_prompt, List):
#             negative_prompt = [negative_prompt] * num_prompts

#         image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)  # {1, 16, 768}
#         bs_embed, seq_len, _ = image_prompt_embeds.shape
#         image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)  # {1, 64, 768}
#         image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
#         uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
#         uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

#         with torch.inference_mode():
#             prompt_embeds = self.pipe._encode_prompt(
#                 prompt, device=self.device, num_images_per_prompt=num_samples, do_classifier_free_guidance=True,
#                 negative_prompt=negative_prompt)
#             negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)  # {1, 77, 768}
#             prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)  # {1, 93, 768}

#             negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

#         generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
#         images = self.pipe(
#             prompt_embeds=prompt_embeds,
#             negative_prompt_embeds=negative_prompt_embeds,
#             guidance_scale=guidance_scale,
#             num_inference_steps=num_inference_steps,
#             generator=generator,
#             **kwargs,
#         ).images

#         return images


# class IPAdapterPlus(IPAdapter):
#     """IPAdapter with fine-grained features"""

#     def init_proj(self):
#         image_proj_model = Resampler(
#             dim=self.pipe.unet.config.cross_attention_dim,
#             depth=4,
#             dim_head=64,
#             heads=12,
#             num_queries=self.num_tokens,
#             embedding_dim=self.image_encoder.config.hidden_size,
#             output_dim=self.pipe.unet.config.cross_attention_dim,
#             ff_mult=4
#         ).to(self.device, dtype=torch.float16)
#         return image_proj_model
    
#     @torch.inference_mode()
#     def get_image_embeds(self, pil_image):
#         if isinstance(pil_image, Image.Image):
#             pil_image = [pil_image]
#         clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
#         clip_image = clip_image.to(self.device, dtype=torch.float16)
#         clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
#         image_prompt_embeds = self.image_proj_model(clip_image_embeds)
#         uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2]
#         uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
#         return image_prompt_embeds, uncond_image_prompt_embeds

