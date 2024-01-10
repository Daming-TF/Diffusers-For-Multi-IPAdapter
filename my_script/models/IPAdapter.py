from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch

from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import EXAMPLE_DOC_STRING, rescale_noise_cfg

# from .base import *
import os
import sys
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
from ip_adapter import IPAdapterPlus
from ip_adapter.attention_processor import AttnProcessor2_0
from my_script.models.multi_ipadapter import MultiIPAttnProcessor2_0


def get_ipadapter_hidden_states(cross_attention_states, encoder_hidden_states_group):
    ip_hidden_states_group = []
    if cross_attention_states=='txt':
        for encoder_hidden_states in encoder_hidden_states_group:
            encoder_hidden_states_ = encoder_hidden_states[0]
            ip_hidden_states_group.append(encoder_hidden_states_)
    elif cross_attention_states=='txt_img':
        for encoder_hidden_states in encoder_hidden_states_group:
            encoder_hidden_states_ = encoder_hidden_states[1]
            ip_hidden_states_group.append(encoder_hidden_states_) 

    else:
        print("cross_attention_kwargs['down_blocks'] is must chosed in ['txt_img', 'txt']")
        exit(0)
    return ip_hidden_states_group


# TODO BY JIAHUI:V2 suport "encoder hidden states" parameter input type is list
#   exp - [fine control- add weights_enblocks]
class UNet2DConditionModelV1(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        # encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        # JIAHUI'S MODIFY
        encoder_hidden_states: Union[List[List[torch.Tensor]], List[torch.Tensor]] = None,
        freeU_kwarge: Optional[Dict[str, Any]] = None,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            print("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

                # `Timesteps` does not contain any weights and will always return f32 tensors
                # there might be better ways to encapsulate this.
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        if self.config.addition_embed_type == "text":       # None
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            # SDXL - style
            if "text_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                )
            text_embeds = added_cond_kwargs.get("text_embeds")
            if "time_ids" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                )
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
                )
            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":       # None
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kadinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = mid_block_additional_residual is None and down_block_additional_residuals is not None

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                # For t2i-adapter CrossAttnDownBlock2D
                additional_residuals = {}
                if is_adapter and len(down_block_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_block_additional_residuals.pop(0)
                
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,       
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    # cross_attention_kwargs={
                    #     # 'mode':cross_attention_kwargs['down_blocks'],
                    #     'weights_enable':cross_attention_kwargs['weights_enable'],
                    #     'cka_dir': cross_attention_kwargs.get("cka_dir", None),
                    #     },
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                if is_adapter and len(down_block_additional_residuals) > 0:
                    sample += down_block_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.mid_block is not None:
            # JIAHUI'S MODIFY
            # # get ipadapter hidden states
            # ip_hidden_states_group = get_ipadapter_hidden_states(cross_attention_kwargs['mid_block'], encoder_hidden_states_group)
            # ip_hidden_states_group = [encoder_hidden_states[1] for encoder_hidden_states in encoder_hidden_states_group]
            # ++++++++++++++++++++++++++++

            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                # cross_attention_kwargs={
                #     # 'mode':cross_attention_kwargs['mid_block'],
                #     'weights_enable':cross_attention_kwargs['weights_enable'],
                #     'cka_dir': cross_attention_kwargs.get("cka_dir", None),
                #     },
                encoder_attention_mask=encoder_attention_mask,
            )

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # --------------- FreeU code -----------------------
            # sample[:,:640]: {2, 640,32,32}
            if freeU_kwarge is not None:
                print('using freeU')
                b1, b2 = freeU_kwarge['b1'], freeU_kwarge['b2']
                s1, s2 = freeU_kwarge['s1'], freeU_kwarge['s2']
                from .freeU import Fourier_filter
                vision = freeU_kwarge['vision']
                assert (vision=='v1') or (vision=='v2')

                res_samples_update = []
                for i, hs_ in enumerate(res_samples):      # {2,1280,8,8}
                    # Only operate on the first two stages
                    if sample.shape[1] == 1280:
                        if vision == 'v1':
                            sample[:,:640] = sample[:,:640] * b1
                            hs_ = Fourier_filter(hs_, threshold=1, scale=s1)
                        elif vision == 'v2':
                            hidden_mean = sample.mean(1).unsqueeze(1)

                            B = hidden_mean.shape[0]
                            hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                            hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                            sample[:, :640] = sample[:, :640] * ((b1 - 1) * hidden_mean + 1)
                            # print( ((b1 - 1) * hidden_mean + 1))

                            hs_ = Fourier_filter(hs_, threshold=1, scale=s1)
                    if sample.shape[1] == 640:
                        if vision == 'v1':
                            sample[:,:320] = sample[:,:320] * b2
                            hs_ = Fourier_filter(hs_, threshold=1, scale=s2)
                        elif vision == 'v2':
                            hidden_mean = sample.mean(1).unsqueeze(1)
                            B = hidden_mean.shape[0]
                            hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                            hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
                            hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)

                            sample[:, :320] = sample[:, :320] * ((b2 - 1) * hidden_mean + 1)
                            hs_ = Fourier_filter(hs_, threshold=1, scale=s2)

                    res_samples_update.append(hs_)
                res_samples = tuple(res_samples_update)
            # ---------------------------------------------------------

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                # JIAHUI'S MODIFY
                # # get ipadapter hidden states
                # ip_hidden_states_group = get_ipadapter_hidden_states(cross_attention_kwargs[f'up_blocks'], encoder_hidden_states_group)
                # ip_hidden_states_group = [encoder_hidden_states[1] for encoder_hidden_states in encoder_hidden_states_group]
                # +++++++++++++++++++++

                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    # cross_attention_kwargs={
                    #     # 'mode': cross_attention_kwargs[f'up_blocks'],
                    #     'weights_enable': cross_attention_kwargs[f'weights_enable'],
                    #     'cka_dir': cross_attention_kwargs.get("cka_dir", None),
                    #     },
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

        # 6. post-process
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

    @property
    def attn_processors(self) -> Dict[str, Optional[Union[AttnProcessor2_0, MultiIPAttnProcessor2_0]]]:
        """
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()        # return_deprecated_lora=True

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors


# TODO BY JIAHUI:V2 'prompt_embeds_groups' ==> Two-dimensional list
# exp - [Single image rotation transform feature averaging]
class StableDiffusionPipelineV1(StableDiffusionPipeline):
    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds_groups: Optional[list] = None,
            negative_prompt_embeds_groups: Optional[list] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds_groups[0][1], negative_prompt_embeds_groups[0][1]
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

        # Jiahui's MODIFICATION multiple prompts.
        prompt_embeds_group_update = []
        for prompt_embed_list, negative_prompt_embed_list in zip(prompt_embeds_groups, negative_prompt_embeds_groups):
            prompt_embeds_list_update = []
            for prompt_embed, negative_prompt_embed in zip(prompt_embed_list, negative_prompt_embed_list):
                prompt_embed = self._encode_prompt(
                    prompt,
                    device,
                    num_images_per_prompt,
                    do_classifier_free_guidance,
                    negative_prompt,
                    prompt_embeds=prompt_embed,
                    negative_prompt_embeds=negative_prompt_embed,
                    lora_scale=text_encoder_lora_scale,
                )
                prompt_embeds_list_update.append(prompt_embed)
            prompt_embeds_group_update.append(prompt_embeds_list_update)
        ####################################################

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
            prompt_embeds_group_update[0][1].dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states_group=prompt_embeds_group_update,        # two dim list # [0][0]: {2, 93 768}
                    cross_attention_kwargs=cross_attention_kwargs,
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
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds_group_update[0][1].dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

# TODO BY JIAHUI:V2 [multi reference image]
# exp - [Single image rotation transform feature averaging]
class IPAdapterV1(IPAdapterPlus):
    def generate(
            self,
            pil_images,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=-1,
            guidance_scale=7.5,
            num_inference_steps=30,
            feature_mode='avg_feature',  # 'simple', 'avg_embeds', 'token_concat', 'avg_feature'
            subtracted_prompts=None,
            subtracted_scale=1,
            **kwargs,
    ):
        self.set_scale(scale)
        num_prompts = 1
        if isinstance(pil_images, Image.Image):
            pil_images = [pil_images]

        # process image prompt embedding
        image_prompt_embeds_list, uncond_image_prompt_embeds_list = [], []
        if feature_mode in ['simple', 'avg_embeds']:
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

        elif feature_mode == 'token_concat':
            cond_embeds_list, uncond_embeds_list = [], []
            for pil_image in pil_images:
                image_prompt_embeds_, uncond_image_prompt_embeds_ = self.get_image_embeds(pil_image)  # {1, 16, 768}
                cond_embeds_list.append(image_prompt_embeds_)
                uncond_embeds_list.append(uncond_image_prompt_embeds_)
            image_prompt_embeds_list = [torch.cat(cond_embeds_list, dim=1)]
            uncond_image_prompt_embeds_list = [torch.cat(uncond_embeds_list, dim=1)]

        elif feature_mode == 'avg_feature':
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
            negative_prompt = "monochrome, lowers, bad anatomy, worst quality, low quality"
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        with torch.inference_mode():
            prompt_embeds = self.pipe._encode_prompt(
                prompt, device=self.device, num_images_per_prompt=num_samples, do_classifier_free_guidance=True,
                negative_prompt=negative_prompt)
            negative_prompt_embeds_, prompt_embeds_ = prompt_embeds.chunk(2)  # {1, 77, 768}

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

                    subtracted_prompt_embeds_, _, = self.pipe.encode_prompt(
                        subtracted_prompt, 
                        device=self.device, 
                        num_images_per_prompt=num_samples, 
                        do_classifier_free_guidance=True,
                        )

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

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        images = self.pipe(
            prompt_embeds_groups=prompt_embeds_groups,
            negative_prompt_embeds_groups=negative_prompt_embeds_groups,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images

# ===================================================================
