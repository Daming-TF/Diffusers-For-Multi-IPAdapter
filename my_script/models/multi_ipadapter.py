from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import os
import torch
from PIL import Image
import numpy as np
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from safetensors.torch import load_file

from insightface.app import FaceAnalysis
from insightface.utils import face_align

from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionXLControlNetPipeline
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers import ControlNetModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import EXAMPLE_DOC_STRING, rescale_noise_cfg, StableDiffusionXLPipelineOutput

import sys
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
from ip_adapter import IPAdapter
from ip_adapter.attention_processor import AttnProcessor2_0, CNAttnProcessor2_0
from ip_adapter.resampler import Resampler
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.ip_adapter_faceid import USE_DAFAULT_ATTN, MLPProjModel, ProjPlusModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available() and (not USE_DAFAULT_ATTN):
    from ip_adapter.attention_processor_faceid import (LoRAAttnProcessor2_0 as LoRAAttnProcessor,)
    from ip_adapter.attention_processor_faceid import (LoRAIPAttnProcessor2_0 as LoRAIPAttnProcessor,)
else:
    from ip_adapter.attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor

from my_script.models.base import MultiIPAttnProcessor2_0
from my_script.util.ui_util import OtherTrick, LoRA

class IPUnit:
    def __init__(self, image_encoder_path, ip_ckpt, num_tokens, device, cross_attention_dim, image_encoder=None):
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        self.device = device
        self.cross_attention_dim = cross_attention_dim
        self.faceid_adapter = os.path.basename(ip_ckpt).split('.')[0] if 'faceid' in self.ip_ckpt else None

        # init image encoder and preprocess
        self.clip_image_processor = CLIPImageProcessor()
        self.init_image_encoder(image_encoder)

        # image proj model
        self.image_proj_model = self.init_proj()
        self.load_ip_adapter()
    
    def load_ip_adapter(self):
        print(f'loading ipadapter...... ==> {self.ip_ckpt}')
        state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.ip_layers = state_dict["ip_adapter"]

    def init_image_encoder(self, image_encoder):
        # load image encoder
        print(f'loading vit...... ==> {self.image_encoder_path}')
        if image_encoder is None and self.image_encoder_path is not None:
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(self.device, dtype=torch.float16)
        else:
            self.image_encoder = image_encoder
        
        if 'faceid' in self.ip_ckpt:
            self.face_encoder = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])    # "buffalo_l"
            self.face_encoder.prepare(ctx_id=0, det_size=(640, 640))

    def init_proj(self):
        if 'plus' in self.ip_ckpt and 'faceid' not in self.ip_ckpt:
            image_proj_model = Resampler(
                dim=1280,
                depth=4,
                dim_head=64,
                heads=20,
                num_queries=self.num_tokens,
                embedding_dim=self.image_encoder.config.hidden_size,
                output_dim=self.cross_attention_dim,
                ff_mult=4
            ).to(self.device, dtype=torch.float16)
        elif 'faceid' in self.ip_ckpt:
            if 'plus' in self.ip_ckpt:
                image_proj_model = ProjPlusModel(
                    cross_attention_dim=self.cross_attention_dim,
                    id_embeddings_dim=512,
                    clip_embeddings_dim=self.image_encoder.config.hidden_size,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
            else:
                image_proj_model = MLPProjModel(
                    cross_attention_dim=self.cross_attention_dim,
                    id_embeddings_dim=512,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        else:
            image_proj_model = ImageProjModel(
                cross_attention_dim=self.cross_attention_dim,
                clip_embeddings_dim=self.image_encoder.config.projection_dim,
                clip_extra_context_tokens=self.num_tokens,
            ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image, **kwargs):
        if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
        uncond_image_cache = kwargs.pop('uncond_image_cache')
        gray_uncond_enable = kwargs.pop('gray_uncond_enable')
        s_scale = kwargs.pop('s_scale')
        assert kwargs == {}

        if 'plus' in self.ip_ckpt and 'faceid' not in self.ip_ckpt:
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image = clip_image.to(self.device, dtype=torch.float16)
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            if gray_uncond_enable:
                gray_img = [pil_image[0].convert('L')]
                gray_clip_image = self.clip_image_processor(images=gray_img, return_tensors="pt").pixel_values.to(self.device, dtype=torch.float16)
                uncond_clip_image_embeds = self.image_encoder(gray_clip_image, output_hidden_states=True).hidden_states[-2]
            else:
                uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2]
            uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        elif 'faceid' in self.ip_ckpt:
            assert len(pil_image)==1, TypeError("input image not support 'List'")
            if isinstance(pil_image[0], Image.Image):
                np_image = np.array(pil_image[0])[:, :, ::-1]
            faces = self.face_encoder.get(np_image)
            assert len(faces) == 1, "The number of faces in the picture is not equal to one"
            faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
            faceid_embeds = faceid_embeds.to(self.device, dtype=torch.float16)
            if 'plus' in self.ip_ckpt:  # for faceid plus v2
                face_image = face_align.norm_crop(np_image, landmark=faces[0].kps, image_size=224) # you can also segment the face
                clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values
                clip_image = clip_image.to(self.device, dtype=torch.float16)
                clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
                uncond_clip_image_embeds = self.image_encoder(
                    torch.zeros_like(clip_image), output_hidden_states=True
                ).hidden_states[-2]
                assert 'v2' in self.ip_ckpt, ValueError("not support faceid plus V1-XL")
                image_prompt_embeds = self.image_proj_model(faceid_embeds, clip_image_embeds, shortcut=True, scale=s_scale)
                uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(faceid_embeds), uncond_clip_image_embeds, shortcut=True, scale=s_scale)
            else:   # for faceid base
                image_prompt_embeds = self.image_proj_model(faceid_embeds)
                uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(faceid_embeds))
        else:
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))

        # reference: https://github.com/lllyasviel/Fooocus/discussions/557
        if uncond_image_cache is not None:
            assert os.path.splitext(uncond_image_cache)[1] == '.safetensors', 'uncondition image embeds must be safetensors file'
            print(f'loading uncond img embeds......\t==>\t{uncond_image_cache}')
            uncond_image_prompt_embeds = load_file(uncond_image_cache)['data'].to(image_prompt_embeds.device)

        return image_prompt_embeds, uncond_image_prompt_embeds


class MultiIpadapter:
    def __init__(self,sd_pipe, units_param, device, ip_count=4):
        self.ip_count = ip_count
        self.units_param = units_param      # image_encoder, ip_ckpt, num_token
        self.device = device
        self.pipe = sd_pipe.to(self.device)
        self.ip_units = [None] * self.ip_count
        self.units_enable = [False] * self.ip_count
        self.units_num_token = [None] * self.ip_count

        self.get_ip_units()

        self.MultiIPAttnProcessor = MultiIPAttnProcessor2_0
        self.load_attn()

    def update_unit(self, unit_id, param_dict):
        self._get_ip_units(unit_id, param_dict)
        self._load_attn(unit_id, self.ip_units[unit_id])

    def _load_attn(self, unit_id, ip_unit):
        '''
            unit_id: Optional[int]
            ip_unit: Optional[IPUnit]}
        '''
        if ip_unit is None: 
            return
        
        atten_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        for layer_id, atten_layer in enumerate(atten_layers):
            if not isinstance(atten_layer, MultiIPAttnProcessor2_0):
                continue 
            atten_layer.to_k_ip[unit_id].load_state_dict({'weight': ip_unit.ip_layers.pop(f'{layer_id}.to_k_ip.weight')})
            atten_layer.to_v_ip[unit_id].load_state_dict({'weight': ip_unit.ip_layers.pop(f'{layer_id}.to_v_ip.weight')})
        # checked model weights
        miss_keys = []
        for k in ip_unit.ip_layers.keys():
            miss_keys.append(k) if 'lora' not in k else None
        assert len(miss_keys)==0, f'Unit{unit_id} weights is missmatching: {ip_unit.ip_layers.keys()}'

    def load_attn(self):
        self.set_atten()
        for u_id, ip_unit in enumerate(self.ip_units):
            self._load_attn(u_id, ip_unit)
    
    def _get_ip_units(self, unit_id, unit_param):
        image_encoder_path = unit_param['image_encoder_path']
        ip_ckpt = unit_param['ip_ckpt']
        num_tokens = unit_param['num_tokens']

        # del faceid lora
        if self.ip_units[unit_id] is not None and self.ip_units[unit_id].faceid_adapter is not None:
            adapter_name = self.ip_units[unit_id].faceid_adapter
            print(f"unloading lora for...... ==> {adapter_name}")
            self.pipe.delete_adapters(adapter_name)
        
        if ip_ckpt is None:
            self.ip_units[unit_id] = None
            self.units_num_token[unit_id] = None
            return 

        print(f"loading the Unit{unit_id}......." )
        image_encoder = None if self.ip_units[unit_id] is None or self.ip_units[unit_id].image_encoder_path != image_encoder_path \
                else self.ip_units[unit_id].image_encoder
        ip_unit = IPUnit(
            image_encoder_path, 
            ip_ckpt, 
            num_tokens, 
            self.device,
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            image_encoder=image_encoder,
            )
        
        # load faceid lora 
        if ip_unit.faceid_adapter is not None:
            adapter_name = ip_unit.faceid_adapter
            print(f"loading lora for...... ==> {adapter_name}")
            lora_path = LoRA.faceid_lora.value[adapter_name]
            self.pipe.load_lora_weights(
                os.path.dirname(lora_path),
                weight_name=os.path.basename(lora_path), 
                adapter_name=adapter_name
                )


        # self.units_enable[unit_id] = True
        self.ip_units[unit_id] = ip_unit
        self.units_num_token[unit_id] = num_tokens
        return

    def get_ip_units(self):
        for unit_id, unit_param in self.units_param.items():
            self._get_ip_units(unit_id, unit_param)

    def set_atten(self):
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
            
            # fix for faceid
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor2_0()
            else:
                attn_procs[name] = self.MultiIPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, units_num=self.ip_count).to(\
                    self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            from .base import CNAttnProcessor2_1
            self.pipe.controlnet.set_attn_processor(CNAttnProcessor2_1())

    def set_atten_param(self, ip_scale, ip_units_enable, ip_kv_norm):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, self.MultiIPAttnProcessor):
                attn_processor.ip_scale = ip_scale
                attn_processor.ip_units_enable = ip_units_enable
                attn_processor.ip_kv_norm = ip_kv_norm
                # attn_processor.reset_block_count()

    def generate(
            self,
            prompt=None,
            negative_prompt=None,
            num_samples=4,
            seed=-1,
            guidance_scale=7.5,
            num_inference_steps=30,
            # units_param
            pil_images: Optional[List[List]]=None,
            ip_scale: Optional[List[List]]=None, # ip_scale
            feature_mode: Optional[list]=None,
            **kwargs,
            ):
        # 0. init
        trick = kwargs.pop('trick')
        cache_paths = kwargs.pop('cache_paths')
        ip_kv_norm = True if OtherTrick.IP_KV_NORM.value in trick else False
        uncond_image_cache = OtherTrick.UNCOND_IMG_EMBEDS_CACHE_PATH.value if OtherTrick.UNCOND_IMG_EMBEDS_CACHE.value in trick else None
        gray_uncond_enable = True if OtherTrick.GRAY_UNCOND_IMG_EMBEDS.value in trick else False
        get_image_embeds_args = {
            'uncond_image_cache': uncond_image_cache, 
            'gray_uncond_enable': gray_uncond_enable,
            's_scale': kwargs.pop('s_scale', 1.0),
            }

        # 1. Set AttenParam
        self.units_enable = [True for _ in pil_images] if self.units_enable.count(True) == 0 else self.units_enable
        self.set_atten_param(ip_scale, self.units_enable, ip_kv_norm)

        # 2. Get ImageEmbeddings 
        enable_unit_id = kwargs.pop('enable_unit_id', [i for i, enable in enumerate(self.units_enable) if enable is True])     # fix api of enhanced vision
        # enable_unit_id = [i for i, enable in enumerate(self.units_enable) if enable is True]

        assert len(enable_unit_id)==len(pil_images)==len(feature_mode)==len(ip_scale)==len(cache_paths)
        # assert len(pil_images)==len(feature_mode)==len(ip_scale)==len(cache_paths)
        units_image_embeds, units_uncond_image_embeds = [], []
        for unit_id, unit_imgs, feature_mode_, ip_scale_, cache_path in zip(enable_unit_id, pil_images, feature_mode, ip_scale, cache_paths):
            if cache_path != '':
                embeds_cache = torch.load(cache_path, map_location=torch.device('cpu'))
                image_embeds = embeds_cache['image_embed'].to(self.pipe.dtype)
                uncond_image_embeds = embeds_cache['uncond_image_embed'].to(self.pipe.dtype)
                if len(image_embeds.shape) == 2 and len(uncond_image_embeds.shape) == 2:
                    image_embeds = image_embeds.unsqueeze(0)
                    uncond_image_embeds = uncond_image_embeds.unsqueeze(0)
                if isinstance(image_embeds, torch.Tensor) and isinstance(uncond_image_embeds, torch.Tensor):
                    image_embeds = [image_embeds]
                    uncond_image_embeds = [uncond_image_embeds]   

            else:
                if feature_mode_ in ['simple', 'avg_embeds']:
                    image_embeds, uncond_image_embeds = None, None
                    assert len(ip_scale_)==1
                    for unit_img in unit_imgs:
                        image_embeds_, uncond_image_embeds_ = self.ip_units[unit_id].get_image_embeds(unit_img, **get_image_embeds_args)  # {1, 16, 768}
                        if image_embeds is None and uncond_image_embeds is None:
                            image_embeds, uncond_image_embeds = image_embeds_, uncond_image_embeds_
                        else:
                            image_embeds = image_embeds.clone() + image_embeds_
                            uncond_image_embeds = uncond_image_embeds.clone() + uncond_image_embeds_
                    
                    image_embeds = [image_embeds / len(unit_imgs)]
                    uncond_image_embeds = [uncond_image_embeds / len(unit_imgs)]
                elif feature_mode_ == 'avg_feature':
                    image_embeds, uncond_image_embeds = [], []
                    assert len(ip_scale_)==len(unit_imgs)
                    for unit_img in unit_imgs:
                        image_embeds_, uncond_image_embeds_ = self.ip_units[unit_id].get_image_embeds(unit_img, **get_image_embeds_args)
                        image_embeds.append(image_embeds_)
                        uncond_image_embeds.append(uncond_image_embeds_)
                else:
                    print("Error: feature mode is not chosed in ['simple', 'avg_embeds', 'avg_feature']")
                    exit(1)
                
            units_image_embeds.append(image_embeds)
            units_uncond_image_embeds.append(uncond_image_embeds)

        # 3. Get Text Embeddings
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        
        units_prompt_embeds, units_uncond_prompt_embeds = [], []
        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
                prompt, num_images_per_prompt=num_samples, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            for image_prompt_embeds, uncond_image_prompt_embeds in zip(units_image_embeds, units_uncond_image_embeds):
                prompt_embeds, negative_prompt_embeds = [], []
                for image_prompt_embeds_, uncond_image_prompt_embeds_ in zip(image_prompt_embeds, uncond_image_prompt_embeds):
                    # (1) when generating multiple graphs at once, the tensor needs to be reshaped
                    bs_embed, seq_len, _ = image_prompt_embeds_.shape

                    image_prompt_embeds_ = image_prompt_embeds_.repeat(1, num_samples, 1)  # {1, 16, 768}
                    image_prompt_embeds_ = image_prompt_embeds_.view(bs_embed * num_samples, seq_len, -1)
                    # units_prompt_embeds[i][j] = image_prompt_embeds_vscode-file://vscode-app/c:/Users/admin/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html

                    uncond_image_prompt_embeds_ = uncond_image_prompt_embeds_.repeat(1, num_samples, 1)
                    uncond_image_prompt_embeds_ = uncond_image_prompt_embeds_.view(bs_embed * num_samples, seq_len, -1)
                    # units_negative_prompt_embeds[i][j] = uncond_image_prompt_embeds_

                    # (2) concat image embeds and txt embeds
                    prompt_embeds__ = torch.cat([prompt_embeds_, image_prompt_embeds_.to(prompt_embeds_.device)], dim=1)
                    negative_prompt_embeds__ = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds_.to(prompt_embeds_.device)], dim=1)

                    prompt_embeds.append(prompt_embeds__)
                    negative_prompt_embeds.append(negative_prompt_embeds__)
                
                units_prompt_embeds.append(prompt_embeds)
                units_uncond_prompt_embeds.append(negative_prompt_embeds)

        # 4. Processing
        # cross_attention_kwargs = kwargs.pop('cross_attention_kwargs')
        # cross_attention_kwargs.update({
        #         'units_num_token': [self.units_num_token[i] for i, enable in enumerate(self.units_enable) if enable is True],
        #         })
        
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # import random
        # import numpy as np
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = True
        # os.environ['PYTHONHASHSEED'] = str(seed)
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        images = self.pipe(
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            # multi-ip param:
            prompt_embeds_groups=units_prompt_embeds,
            negative_prompt_embeds_groups=units_uncond_prompt_embeds,
            **kwargs,
        ).images
 
        return images
