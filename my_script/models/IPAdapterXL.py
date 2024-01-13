from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch

from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils.doc_utils import replace_example_docstring
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import EXAMPLE_DOC_STRING, rescale_noise_cfg, StableDiffusionXLPipelineOutput
from diffusers.image_processor import PipelineImageInput

from diffusers import StableDiffusionXLControlNetPipeline
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers import ControlNetModel

# from .base import IPAdapter, Resampler
# from util.ui_util import ControlMode
from ip_adapter.resampler import Resampler
from ip_adapter import IPAdapter
from my_script.util.ui_util import ControlMode


class IPAdapterXL(IPAdapter):
    """SDXL"""
    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=-1,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)
        
        if isinstance(pil_image, Image.Image):
            num_prompts = 1
        else:
            num_prompts = len(pil_image)
        
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowers, bad anatomy, worst quality, low quality"
            
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
                prompt, num_images_per_prompt=num_samples, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
            
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
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
    

class IPAdapterPlusXL(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4
        ).to(self.device, dtype=torch.float16)
        return image_proj_model
    
    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    def generate(
        self,
        pil_image,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=-1,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)
        
        if isinstance(pil_image, Image.Image):
            num_prompts = 1
        else:
            num_prompts = len(pil_image)
        
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
                prompt, num_images_per_prompt=num_samples, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
            
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
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
    

# TODO BY JIAHUI:V1 
#   exp - [fine control- add weights_enblocks]
#   exp - [subtracted_feature_from_embedding]
class IPAdapterPlusXLV1(IPAdapter):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4
        ).to(self.device, dtype=torch.float16)
        return image_proj_model
    
    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    def generate(
        self,
        pil_images,
        prompt=None,
        negative_prompt=None,
        scale=1.0,
        num_samples=4,
        seed=-1,
        num_inference_steps=30,
        feature_mode='avg_feature',
        subtracted_prompts=None,
        subtracted_scale=1,
        **kwargs,
    ):
        scale = scale if isinstance(scale, list) else [scale]
        self.set_scale(scale)
        num_prompts = 1
        if isinstance(pil_images, Image.Image):
            pil_images = [pil_images]
        
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        
        image_prompt_embeds_list, uncond_image_prompt_embeds_list = [], []
        if feature_mode in ['simple', 'avg_embeds']:
            assert len(scale)==1, f"feature mode choise in ['simple', 'avg_embeds'], so scale just 1 element ==> {len(scale)}"
            image_prompt_embeds, uncond_image_prompt_embeds = None, None
            for pil_image in pil_images:
                image_prompt_embeds_, uncond_image_prompt_embeds_ = self.get_image_embeds(pil_image)  # {1, 16, 768}
                if image_prompt_embeds is None and uncond_image_prompt_embeds is None:
                    image_prompt_embeds = image_prompt_embeds_
                    uncond_image_prompt_embeds = uncond_image_prompt_embeds_
                else:
                    image_prompt_embeds = image_prompt_embeds.clone() + image_prompt_embeds_
                    uncond_image_prompt_embeds = uncond_image_prompt_embeds.clone() + uncond_image_prompt_embeds_
            image_prompt_embeds_list = [image_prompt_embeds / len(pil_images)]
            uncond_image_prompt_embeds_list = [uncond_image_prompt_embeds / len(pil_images)]
       
        # elif feature_mode == 'token_concat':
        #     cond_embeds_list, uncond_embeds_list = [], []
        #     for pil_image in pil_images:
        #         image_prompt_embeds_, uncond_image_prompt_embeds_ = self.get_image_embeds(pil_image)  # {1, 16, 768}
        #  1      cond_embeds_list.append(image_prompt_embeds_)
        #         uncond_embeds_list.append(uncond_image_prompt_embeds_)
        #     image_prompt_embeds_list = [torch.cat(cond_embeds_list, dim=1)]
        #     uncond_image_prompt_embeds_list = [torch.cat(uncond_embeds_list, dim=1)]

        elif feature_mode == 'avg_feature':     # after cross atten
            assert len(scale)==len(pil_images), f"feature mode: avg_feature, so scale just 1 element ==> {len(scale)}"
            for pil_image in pil_images:
                image_prompt_embeds_, uncond_image_prompt_embeds_ = self.get_image_embeds(pil_image)
                image_prompt_embeds_list.append(image_prompt_embeds_)
                uncond_image_prompt_embeds_list.append(uncond_image_prompt_embeds_)
        else:
            print(f"feature mode is not chosed in ['simple', 'avg_embeds', 'token_concat']")
            exit(1)

        for index, (image_prompt_embeds, uncond_image_prompt_embeds) in \
                enumerate(zip(image_prompt_embeds_list, uncond_image_prompt_embeds_list)):
            bs_embed, seq_len, _ = image_prompt_embeds.shape

            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)  # {1, 16, 768}
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            image_prompt_embeds_list[index] = image_prompt_embeds

            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
            uncond_image_prompt_embeds_list[index] = uncond_image_prompt_embeds

        # process txt prompt embedding
        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
                prompt, num_images_per_prompt=num_samples, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            # prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            # negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

            # JIAHUI'S MODIFICATION: SUBTRACTED FEATURE FROM EMBEDDING 2023.10.10
            if subtracted_prompts is None:
                subtracted_prompts_embeds = [torch.zeros_like(image_prompt_embeds)]*len(image_prompt_embeds_list)
                # pooled_subtracted_prompts_emdeds = subtracted_prompts_embeds
            else:
                if isinstance(subtracted_prompts, str):
                    subtracted_prompts = [subtracted_prompts]
                assert isinstance(subtracted_prompts, list) and len(subtracted_prompts)==len(pil_images)
                subtracted_prompts_embeds = []
                for i, subtracted_prompt in enumerate(subtracted_prompts):
                    subtracted_prompt = subtracted_prompt if self.blip is None else self.blip(pil_images[i])
                    print(f"\033[91m 'subtracted_prompts' is ==> {subtracted_prompt} \033[0m")

                    subtracted_prompt_embeds_, _, _, _ = self.pipe.encode_prompt(subtracted_prompt)
                    null_embeds_, _, _, _ = self.pipe.encode_prompt('')

                    # # M1:
                    # image_prompt_embeds = image_prompt_embeds_list[i]
                    # # trans_scale = torch.sqrt(image_prompt_embeds.var()) / torch.sqrt(subtracted_prompt_embeds_[:, :16, :].var())
                    # # new_subtracted_prompts_embed = (subtracted_prompt_embeds_[:, :16, :] - subtracted_prompt_embeds_[:, :16, :].mean()) * trans_scale + image_prompt_embeds.mean()*0.1
                    # subtracted_prompts_embeds.append(new_subtracted_prompts_embed)
                    
                    # # M2:
                    # diff_embeds = subtracted_prompt_embeds_ - null_embeds_
                    # trans_scale = torch.sqrt(image_prompt_embeds.var()) / torch.sqrt(diff_embeds[:, :16, :].var())
                    # new_subtracted_prompts_embed = (diff_embeds[:, :16, :] - diff_embeds[:, :16, :].mean()) * trans_scale + image_prompt_embeds.mean()*0.1
                    # subtracted_prompts_embeds.append(new_subtracted_prompts_embed)
                    # print(f"\033[91m scale: {subtracted_scale} \033[0m")
                    # subtracted_prompts_embeds.append(diff_embeds[:, :16, :]*subtracted_scale)

                    # M3:
                    print(f"\033[91m scale: {subtracted_scale} \033[0m")
                    subtracted_prompts_embeds.append((prompt_embeds_-subtracted_prompt_embeds_)[:, :16, :]*subtracted_scale)

            ##########################################################

            # JIAHUI'S MODIFICATION: SUPPORT MULTIPLE REFERENCE IMAGES
            prompt_embeds_groups, negative_prompt_embeds_groups = [], []
            for i, (image_prompt_embeds, uncond_image_prompt_embeds) in enumerate(zip(image_prompt_embeds_list, uncond_image_prompt_embeds_list)):
                # subtracted embeds
                subtracted_prompts_embed = subtracted_prompts_embeds[i]

                prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds+subtracted_prompts_embed], dim=1)  # {1, 93, 768}
                negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

                # for upblocks only
                prompt_embeds_list = [prompt_embeds_, prompt_embeds]
                negative_prompt_embeds_list = [negative_prompt_embeds_, negative_prompt_embeds]

                # for avg feature
                prompt_embeds_groups.append(prompt_embeds_list)
                negative_prompt_embeds_groups.append(negative_prompt_embeds_list)
            ##############################################################
        generator = torch.Generator(self.device).manual_seed(seed) if (seed != -1) and (seed is not None) else None
        images = self.pipe(
            prompt_embeds_groups=prompt_embeds_groups,
            negative_prompt_embeds_groups=negative_prompt_embeds_groups,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images
        
        return images


class IPAdapterXLV1(IPAdapter):
    def generate(
            self,
            pil_images,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=-1,
            num_inference_steps=30,
            feature_mode='avg_feature',
            subtracted_prompts=None,
            subtracted_scale=1,
            **kwargs,
        ):
            scale = scale if isinstance(scale, list) else [scale]
            self.set_scale(scale)
            num_prompts = 1
            if isinstance(pil_images, Image.Image):
                pil_images = [pil_images]
            
            if prompt is None:
                prompt = "best quality, high quality"
            if negative_prompt is None:
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                
            if not isinstance(prompt, List):
                prompt = [prompt] * num_prompts
            if not isinstance(negative_prompt, List):
                negative_prompt = [negative_prompt] * num_prompts
            
            image_prompt_embeds_list, uncond_image_prompt_embeds_list = [], []
            if feature_mode in ['simple', 'avg_embeds']:
                assert len(scale)==1, f"feature mode choise in ['simple', 'avg_embeds'], so scale just 1 element ==> {len(scale)}"
                image_prompt_embeds, uncond_image_prompt_embeds = None, None
                for pil_image in pil_images:
                    image_prompt_embeds_, uncond_image_prompt_embeds_ = self.get_image_embeds(pil_image)  # {1, 16, 768}
                    if image_prompt_embeds is None and uncond_image_prompt_embeds is None:
                        image_prompt_embeds = image_prompt_embeds_
                        uncond_image_prompt_embeds = uncond_image_prompt_embeds_
                    else:
                        image_prompt_embeds = image_prompt_embeds.clone() + image_prompt_embeds_
                        uncond_image_prompt_embeds = uncond_image_prompt_embeds.clone() + uncond_image_prompt_embeds_
                image_prompt_embeds_list = [image_prompt_embeds / len(pil_images)]
                uncond_image_prompt_embeds_list = [uncond_image_prompt_embeds / len(pil_images)]
        
            # elif feature_mode == 'token_concat':
            #     cond_embeds_list, uncond_embeds_list = [], []
            #     for pil_image in pil_images:
            #         image_prompt_embeds_, uncond_image_prompt_embeds_ = self.get_image_embeds(pil_image)  # {1, 16, 768}
            #  1      cond_embeds_list.append(image_prompt_embeds_)
            #         uncond_embeds_list.append(uncond_image_prompt_embeds_)
            #     image_prompt_embeds_list = [torch.cat(cond_embeds_list, dim=1)]
            #     uncond_image_prompt_embeds_list = [torch.cat(uncond_embeds_list, dim=1)]

            elif feature_mode == 'avg_feature':     # after cross atten
                assert len(scale)==len(pil_images), f"feature mode: avg_feature, so scale just 1 element ==> {len(scale)}"
                for pil_image in pil_images:
                    image_prompt_embeds_, uncond_image_prompt_embeds_ = self.get_image_embeds(pil_image)
                    image_prompt_embeds_list.append(image_prompt_embeds_)
                    uncond_image_prompt_embeds_list.append(uncond_image_prompt_embeds_)
            else:
                print(f"feature mode is not chosed in ['simple', 'avg_embeds', 'token_concat']")
                exit(1)

            for index, (image_prompt_embeds, uncond_image_prompt_embeds) in \
                    enumerate(zip(image_prompt_embeds_list, uncond_image_prompt_embeds_list)):
                bs_embed, seq_len, _ = image_prompt_embeds.shape

                image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)  # {1, 16, 768}
                image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
                image_prompt_embeds_list[index] = image_prompt_embeds

                uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
                uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
                uncond_image_prompt_embeds_list[index] = uncond_image_prompt_embeds

            # process txt prompt embedding
            if prompt is None:
                prompt = "best quality, high quality"
            if negative_prompt is None:
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            if not isinstance(prompt, List):
                prompt = [prompt] * num_prompts
            if not isinstance(negative_prompt, List):
                negative_prompt = [negative_prompt] * num_prompts

            with torch.inference_mode():
                prompt_embeds_, negative_prompt_embeds_, pooled_prompt_embeds, negative_pooled_prompt_embeds = self.pipe.encode_prompt(
                    prompt, num_images_per_prompt=num_samples, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
                # prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
                # negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

                # JIAHUI'S MODIFICATION: SUBTRACTED FEATURE FROM EMBEDDING 2023.10.10
                if subtracted_prompts is None:
                    subtracted_prompts_embeds = [torch.zeros_like(image_prompt_embeds)]*len(image_prompt_embeds_list)
                    # pooled_subtracted_prompts_emdeds = subtracted_prompts_embeds
                else:
                    if isinstance(subtracted_prompts, str):
                        subtracted_prompts = [subtracted_prompts]
                    assert isinstance(subtracted_prompts, list) and len(subtracted_prompts)==len(pil_images)
                    subtracted_prompts_embeds = []
                    for i, subtracted_prompt in enumerate(subtracted_prompts):
                        subtracted_prompt = subtracted_prompt if self.blip is None else self.blip(pil_images[i])
                        print(f"\033[91m 'subtracted_prompts' is ==> {subtracted_prompt} \033[0m")

                        subtracted_prompt_embeds_, _, _, _ = self.pipe.encode_prompt(subtracted_prompt)
                        null_embeds_, _, _, _ = self.pipe.encode_prompt('')

                        # # M1:
                        # image_prompt_embeds = image_prompt_embeds_list[i]
                        # # trans_scale = torch.sqrt(image_prompt_embeds.var()) / torch.sqrt(subtracted_prompt_embeds_[:, :16, :].var())
                        # # new_subtracted_prompts_embed = (subtracted_prompt_embeds_[:, :16, :] - subtracted_prompt_embeds_[:, :16, :].mean()) * trans_scale + image_prompt_embeds.mean()*0.1
                        # subtracted_prompts_embeds.append(new_subtracted_prompts_embed)
                        
                        # # M2:
                        # diff_embeds = subtracted_prompt_embeds_ - null_embeds_
                        # trans_scale = torch.sqrt(image_prompt_embeds.var()) / torch.sqrt(diff_embeds[:, :16, :].var())
                        # new_subtracted_prompts_embed = (diff_embeds[:, :16, :] - diff_embeds[:, :16, :].mean()) * trans_scale + image_prompt_embeds.mean()*0.1
                        # subtracted_prompts_embeds.append(new_subtracted_prompts_embed)
                        # print(f"\033[91m scale: {subtracted_scale} \033[0m")
                        # subtracted_prompts_embeds.append(diff_embeds[:, :16, :]*subtracted_scale)

                        # M3:
                        print(f"\033[91m scale: {subtracted_scale} \033[0m")
                        subtracted_prompts_embeds.append((prompt_embeds_-subtracted_prompt_embeds_)[:, :16, :]*subtracted_scale)

                ##########################################################

                # JIAHUI'S MODIFICATION: SUPPORT MULTIPLE REFERENCE IMAGES
                prompt_embeds_groups, negative_prompt_embeds_groups = [], []
                for i, (image_prompt_embeds, uncond_image_prompt_embeds) in enumerate(zip(image_prompt_embeds_list, uncond_image_prompt_embeds_list)):
                    # subtracted embeds
                    subtracted_prompts_embed = subtracted_prompts_embeds[i]

                    prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds+subtracted_prompts_embed], dim=1)  # {1, 93, 768}
                    negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

                    # for upblocks only
                    prompt_embeds_list = [prompt_embeds_, prompt_embeds]
                    negative_prompt_embeds_list = [negative_prompt_embeds_, negative_prompt_embeds]

                    # for avg feature
                    prompt_embeds_groups.append(prompt_embeds_list)
                    negative_prompt_embeds_groups.append(negative_prompt_embeds_list)
                ##############################################################
            generator = torch.Generator(self.device).manual_seed(seed) if (seed != -1) and (seed is not None) else None
            images = self.pipe(
                prompt_embeds_groups=prompt_embeds_groups,
                negative_prompt_embeds_groups=negative_prompt_embeds_groups,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=num_inference_steps,
                generator=generator,
                **kwargs,
            ).images
            
            return images


# TODO BY JIAHUI:V1 
#   exp - [fine control- add weights_enblocks]
class StableDiffusionXLPipelineV1(StableDiffusionXLPipeline):
    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        # prompt_embeds: Optional[torch.FloatTensor] = None,
        # negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        # JIAHUI'S MODIFY
        prompt_embeds_groups: Optional[List[List[torch.tensor]]] = None,
        negative_prompt_embeds_groups: Optional[List] = None,
        denoise_control: Optional[List[Dict]] = None,
        **kwargs,
    ):
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds_groups[0][0],
            negative_prompt_embeds_groups[0][0],
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds_groups[0][0].shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        # JIAHUI'S MODIFY
        new_prompt_embeds_groups, new_negative_prompt_embeds_groups = [] , []
        for prompt_embed_list, negative_prompt_embed_list in zip(prompt_embeds_groups, negative_prompt_embeds_groups):
            new_prompt_embeds_list, new_negative_prompt_embeds_list = [], []
            for prompt_embeds, negative_prompt_embeds in zip(prompt_embed_list, negative_prompt_embed_list):
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    negative_prompt=negative_prompt,
                    negative_prompt_2=negative_prompt_2,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    lora_scale=text_encoder_lora_scale,
                    clip_skip=clip_skip,
                )
                new_prompt_embeds_list.append(prompt_embeds)
                new_negative_prompt_embeds_list.append(negative_prompt_embeds)
            new_prompt_embeds_groups.append(new_prompt_embeds_list)
            new_negative_prompt_embeds_groups.append(new_negative_prompt_embeds_list)
        # +++++++++++++++++++++++++++++++++++++++++++++++++

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            new_prompt_embeds_groups[0][0].dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        add_time_ids = self._get_add_time_ids(
            original_size, 
            crops_coords_top_left, 
            target_size, 
            dtype=new_prompt_embeds_groups[0][0].dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=new_prompt_embeds_groups[0][0].dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        # JIAHUI'S MODIFY
        if do_classifier_free_guidance:
            for idx, (new_prompt_embeds_list, new_negative_prompt_embeds_list) in enumerate(zip(new_prompt_embeds_groups, new_negative_prompt_embeds_groups)):
                for i, (prompt_embeds, negative_prompt_embeds) in enumerate(zip(new_prompt_embeds_list, new_negative_prompt_embeds_list)):
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    prompt_embeds = prompt_embeds.to(device)
                    new_prompt_embeds_groups[idx][i] = prompt_embeds

            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # # JIAHUI'S MODIFY
        # # for debug: finging diff between image embeds and text embeds
        # from my_script.util.util import LayerStatus
        # layer_status = LayerStatus()
        # # ++++++++++++++++++
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            from .base import BLOCKS_COUNT
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                # JIAHUI'S MODIFY 
                current_step = 1.0 - t / 1000.0
                assert len(new_prompt_embeds_groups) == len(denoise_control)
                denoise_control_enable = []
                for control_step in denoise_control:
                    denoise_enable = True if current_step >= control_step['s_step'] and current_step <= control_step['e_step'] else False
                    denoise_control_enable.append(denoise_enable)
                cross_attention_kwargs['denoise_control_enable'] = denoise_control_enable

                # # JIAHUI'S MODIFY
                # # Statistical network characteristics
                # layer_status.set_step(t) 
                # cross_attention_kwargs['layer_status']=layer_status
                # # +++++++++++++++

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=new_prompt_embeds_groups,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    **kwargs,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        # # JIAHUI'S MODIFY    
        # layer_status.save(r'./data/layers_status.json')
        # # ++++++++++++++++++++++++++

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


# # TODO BY JIAHUI:V2
# #   exp - [multi ipadater] 
# class StableDiffusionXLPipelineV2(StableDiffusionXLPipeline):
    

# TODO By JIAHUI:V1
class StableDiffusionXLImg2ImgPipelineV1(StableDiffusionXLImg2ImgPipeline):
    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        strength: float = 0.3,
        num_inference_steps: int = 50,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds_groups: Optional[List] = None,
        negative_prompt_embeds_groups: Optional[List] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
    ):
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            strength,
            num_inference_steps,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds_groups[0][1],
            negative_prompt_embeds_groups[0][1],
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds_groups[0][1].shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        # JIAHUI'S MODIFY
        new_prompt_embeds_groups, new_negative_prompt_embeds_groups = [] , []
        for prompt_embed_list, negative_prompt_embed_list in zip(prompt_embeds_groups, negative_prompt_embeds_groups):
            new_prompt_embeds_list, new_negative_prompt_embeds_list = [], []
            for prompt_embeds, negative_prompt_embeds in zip(prompt_embed_list, negative_prompt_embed_list):
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    negative_prompt=negative_prompt,
                    negative_prompt_2=negative_prompt_2,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    lora_scale=text_encoder_lora_scale,
                    clip_skip=clip_skip,
                )
                new_prompt_embeds_list.append(prompt_embeds)
                new_negative_prompt_embeds_list.append(negative_prompt_embeds)
            new_prompt_embeds_groups.append(new_prompt_embeds_list)
            new_negative_prompt_embeds_groups.append(new_negative_prompt_embeds_list)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)

        # 5. Prepare timesteps
        def denoising_value_valid(dnv):
            return isinstance(denoising_end, float) and 0 < dnv < 1

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device, denoising_start=denoising_start if denoising_value_valid else None
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        add_noise = True if denoising_start is None else False
        # 6. Prepare latent variables
        latents = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            new_prompt_embeds_groups[0][1].dtype,
            device,
            generator,
            add_noise,
        )
        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 8. Prepare added time ids & embeddings
        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size

        add_text_embeds = pooled_prompt_embeds
        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=new_prompt_embeds_groups[0][1].dtype,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        # JIAHUI'S MODIFY 2023.10.12
        if do_classifier_free_guidance:
            for idx, (new_prompt_embeds_list, new_negative_prompt_embeds_list) in enumerate(zip(new_prompt_embeds_groups, new_negative_prompt_embeds_groups)):
                for i, (prompt_embeds, negative_prompt_embeds) in enumerate(zip(new_prompt_embeds_list, new_negative_prompt_embeds_list)):
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    prompt_embeds = prompt_embeds.to(device)
                    new_prompt_embeds_groups[idx][i] = prompt_embeds
            
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_text_embeds = add_text_embeds.to(device)

            add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)
            add_time_ids = add_time_ids.to(device)
        # +++++++++++++++++++++++++++++++++++++++++++++++

        # 9. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 9.1 Apply denoising_end
        if (
            denoising_end is not None
            and denoising_start is not None
            and denoising_value_valid(denoising_end)
            and denoising_value_valid(denoising_start)
            and denoising_start >= denoising_end
        ):
            raise ValueError(
                f"`denoising_start`: {denoising_start} cannot be larger than or equal to `denoising_end`: "
                + f" {denoising_end} when using type float."
            )
        elif denoising_end is not None and denoising_value_valid(denoising_end):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states_group=new_prompt_embeds_groups,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


# TODO By JIAHUI:V1
class StableDiffusionXLControlNetPipelineV1(StableDiffusionXLControlNetPipeline):
    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        # prompt_embeds: Optional[torch.FloatTensor] = None,
        # negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        # JIAHUI'S MODIFY
        prompt_embeds_groups: Optional[List] = None,
        negative_prompt_embeds_groups: Optional[List] = None,
        control_mode: Optional[ControlMode] = ControlMode.BALANCED,
        denoise_control: Optional[List[Dict]] = None,
        **kwargs,
    ):
        # assert len(prompt_embeds_groups)==len(negative_prompt_embeds_groups)==1, 'sdxl ipadapter layer-wise control only support for single reference image'
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            image,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds_groups[0][0],
            negative_prompt_embeds_groups[0][0],
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds_groups[0][0].shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        # JIAHUI'S MODIFY
        new_prompt_embeds_groups, new_negative_prompt_embeds_groups = [] , []
        for prompt_embed_list, negative_prompt_embed_list in zip(prompt_embeds_groups, negative_prompt_embeds_groups):
            new_prompt_embeds_list, new_negative_prompt_embeds_list = [], []
            for prompt_embeds, negative_prompt_embeds in zip(prompt_embed_list, negative_prompt_embed_list):
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    negative_prompt=negative_prompt,
                    negative_prompt_2=negative_prompt_2,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    lora_scale=text_encoder_lora_scale,
                    clip_skip=clip_skip,
                )
                new_prompt_embeds_list.append(prompt_embeds)
                new_negative_prompt_embeds_list.append(negative_prompt_embeds)
            new_prompt_embeds_groups.append(new_prompt_embeds_list)
            new_negative_prompt_embeds_groups.append(new_negative_prompt_embeds_list)
        # +++++++++++++++++++++++++++++++++++++++++++++++++

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds_groups[0][0].dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 7.2 Prepare added time ids & embeddings
        if isinstance(image, list):
            original_size = original_size or image[0].shape[-2:]
        else:
            original_size = original_size or image.shape[-2:]
        target_size = target_size or (height, width)

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim
        add_time_ids = self._get_add_time_ids(
            original_size, 
            crops_coords_top_left, 
            target_size,
            dtype=prompt_embeds_groups[0][0].dtype,
            text_encoder_projection_dim=text_encoder_projection_dim
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds_groups[0][0].dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        # JIAHUI'S MODIFY
        if do_classifier_free_guidance:
            for idx, (new_prompt_embeds_list, new_negative_prompt_embeds_list) in enumerate(zip(new_prompt_embeds_groups, new_negative_prompt_embeds_groups)):
                for i, (prompt_embeds, negative_prompt_embeds) in enumerate(zip(new_prompt_embeds_list, new_negative_prompt_embeds_list)):
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    prompt_embeds = prompt_embeds.to(device)
                    new_prompt_embeds_groups[idx][i] = prompt_embeds

            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

            add_text_embeds = add_text_embeds.to(device)
            add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                # JIAHUI'S MODIFY 
                # for image prompt denoising control
                current_step = 1.0 - t / 1000.0
                assert len(new_prompt_embeds_groups) == len(denoise_control)
                denoise_control_enable = []
                for control_step in denoise_control:
                    denoise_enable = True if current_step >= control_step['s_step'] and current_step <= control_step['e_step'] else False
                    denoise_control_enable.append(denoise_enable)
                cross_attention_kwargs['denoise_control_enable'] = denoise_control_enable
                # +++++++++++++++

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = new_prompt_embeds_groups[0][0].chunk(2)[1]       # prompt_embeds.chunk(2)[1]
                    controlnet_added_cond_kwargs = {
                        "text_embeds": add_text_embeds.chunk(2)[1],
                        "time_ids": add_time_ids.chunk(2)[1],
                    }
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = new_prompt_embeds_groups[0][0]       # prompt_embeds
                    controlnet_added_cond_kwargs = added_cond_kwargs

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    added_cond_kwargs=controlnet_added_cond_kwargs,
                    return_dict=False,
                )

                # JIAHUI'S MODIFY
                # Reference: https://github.com/Mikubill/sd-webui-controlnet/blob/main/scripts/hook.py#L589
                if control_mode == getattr(ControlMode.PROMPT, 'value'):
                    down_block_res_samples = [feature * (0.825 ** float(12 - i)) for i, feature in enumerate(down_block_res_samples)]
                    mid_block_res_sample = mid_block_res_sample * (0.825 ** float(12 - len(down_block_res_samples)))
                elif control_mode == getattr(ControlMode.BALANCED, 'value'):
                    down_block_res_samples = down_block_res_samples
                    mid_block_res_sample = mid_block_res_sample
                else:
                    print("Only support two control mode : (1.'Balanced'; (2.'Prompt is more import'")
                    print(f'control_mode:{control_mode}')
                    exit(1)
                # ++++++++++++++++++++++++++++++

                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=new_prompt_embeds_groups,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    **kwargs,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # manually for max memory savings
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

