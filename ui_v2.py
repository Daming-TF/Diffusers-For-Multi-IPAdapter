import gradio as gr
from PIL import Image
import cv2
import numpy as np
import sys
import os
import datetime
import torch
from safetensors.torch import load_file
# from models import load_model
from diffusers import DDIMScheduler

from my_script.util.ui_util import set_parser, get_depths, get_lineart, get_canny
from my_script.util.ui_util import IPAdapterUi, ControlMode, ImageOperation, OtherTrick, LoRA, UiSymbol



def load_model(
        base_model_path, 
        image_encoder_path=None, 
        ip_ckpt=None,
        controlnet_model_path=None, 
        device='cuda',      # cuda
        unet_load=False,
        ):
    global ip_model
    global noise_scheduler
    global controlnet
    global pipe

    global LAYER_NUM
    LAYER_NUM = 70 if 'xl' in base_model_path else 16

    if not 'xl' in base_model_path:
        print('\033[91m Controlnet is only support for sdxl, if you want to use ipadapter layer-wise control + controlnet please switch to sdxl model \033[0m')
        exit(0)

    # 1.init scheduler, SD-1.5 only works on DDIM, while SDXL works while on DPM++ and default sampler
    if noise_scheduler is None:
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
    from my_script.models.IPAdapter import UNet2DConditionModelV1 as UNet2DConditionModel
    if unet_load is True:       #  and unet is None
        print(f'loading unet...... ==> {base_model_path}')
        unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder='unet',
        ).to(dtype=torch.float16)

    if pipe is None:
        if controlnet_model_path is not None:
                print('\033[91m Using controlnet..... \033[0m')
                print(f'loading controlnet...... ==> {controlnet_model_path}')
                from diffusers import ControlNetModel
                from my_script.models.IPAdapterXL import StableDiffusionXLControlNetPipelineV1 as StableDiffusionXLControlNetPipeline
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
            print(f'loading sd-txt2img...... ==> {base_model_path}')        
            from my_script.models.IPAdapterXL import StableDiffusionXLPipelineV1
            pipe = StableDiffusionXLPipelineV1.from_pretrained(
                base_model_path, 
                torch_dtype=torch.float16, 
                add_watermarker=False,
                unet=unet,)
        # pipe.enable_model_cpu_offload()
    else:
        if controlnet_model_path is not None:
            print('\033[91m Using controlnet..... \033[0m')
            print(f'loading controlnet...... ==> {controlnet_model_path}')
            from diffusers import ControlNetModel
            from my_script.models.IPAdapterXL import StableDiffusionXLControlNetPipelineV1 as StableDiffusionXLControlNetPipeline
            controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
            print(f'loading sd...... ==> {base_model_path}')
            pipe = StableDiffusionXLControlNetPipeline(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                text_encoder_2=pipe.text_encoder_2,
                tokenizer=pipe.tokenizer,
                tokenizer_2=pipe.tokenizer_2,
                unet=pipe.unet,
                scheduler=noise_scheduler,
                controlnet=controlnet,
            )
        else:
            if controlnet is not None:
                del controlnet
            print(f'loading sd-txt2img...... ==> {base_model_path}')        
            from my_script.models.IPAdapterXL import StableDiffusionXLPipelineV1
            pipe = StableDiffusionXLPipelineV1(
                vae=pipe.vae,
                text_encoder=pipe.text_encoder,
                text_encoder_2=pipe.text_encoder_2,
                tokenizer=pipe.tokenizer,
                tokenizer_2=pipe.tokenizer_2,
                unet=pipe.unet,
                scheduler=noise_scheduler,
            )
        # pipe.enable_model_cpu_offload()

    # 8.load multi ip-adapter
    if image_encoder_path is not None and ip_ckpt is not None:
        print(f'loading ipadapter ..... ==> {ip_ckpt}')
        image_encoder_path = image_encoder_path if isinstance(image_encoder_path, list) else [image_encoder_path]
        ip_ckpt = ip_ckpt if isinstance(ip_ckpt, list) else [ip_ckpt]
        assert len(image_encoder_path)==len(ip_ckpt), \
            f"the length of 'image_encoder_path':{len(image_encoder_path)} and 'ip_ckpt':{len(ip_ckpt)} is different"

        units_parm = {}
        for i, (image_encoder_path_, ip_ckpt_) in enumerate(zip(image_encoder_path, ip_ckpt)):
            units_parm[i] = {
                'image_encoder_path': image_encoder_path_,
                'ip_ckpt': ip_ckpt_,
                'num_tokens': 16 if 'plus' in ip_ckpt_ else 4,
            }

        from my_script.models.multi_ipadapter import MultiIpadapter
        ip_model = MultiIpadapter(pipe, units_parm, device)
    elif image_encoder_path is None and ip_ckpt is None:
        print(f"update pipe to ..... ==> {controlnet_model_path if controlnet_model_path is not None else 'sdxl'}")
        ip_model.update_pipe(pipe)
    else:
        print("** Error: The input type of 'image_encoder_path' and 'ip_ckpt' must be the same!!")
        exit(1)
    print('** loading finish!!')


def get_save_name():
    current_datatime = datetime.datetime.now()
    hour = current_datatime.hour
    minute = current_datatime.minute
    second = current_datatime.second
    return f'{hour}-{minute}-{second}.jpg'


def check_pipe(state):
    global pipe_state
    global lora_state
    global haved_load_ti
    global lora_group

    # 1. update enable status
    for unit_key in list(state.keys()):
        if unit_key in ['base', 'controlnet', 'xy', 'lora']:
            continue
        ip_unit_id = int(unit_key.split('Unit')[1])
        ip_unit_single_enable = state[unit_key]['single_enable']
        ip_unit_multi_enable = state[unit_key]['multi_enable']
        assert not (ip_unit_single_enable is True and ip_unit_multi_enable is True), \
            "single mode and multi mode can't be opened at the same time"
        ip_unit_enable = ip_unit_single_enable or ip_unit_multi_enable

        ip_model.units_enable[ip_unit_id] = ip_unit_enable
        if not ip_unit_enable:
            state.pop(unit_key)

    # 2. update pipe
    control_type = state['controlnet']['control_type']
    if state['controlnet']['is_control']:
        if pipe_state != control_type:
            load_model(
                base_model_path=args.base_model_path,
                controlnet_model_path=controlnet_mode[control_type],
                )
            pipe_state = control_type
    else:
        if pipe_state != 'base':
            load_model(
                base_model_path=args.base_model_path,
                )
            pipe_state = 'base'
    
    # 3. load lora
    lora_id = state['lora'].pop('lora_id')
    if lora_id != lora_state:
        if lora_id is not None:
            # print(f"loading lora...... ==> {getattr(LoRA, lora_id).value}")
            # pipe.load_lora_weights(getattr(LoRA, lora_id).value)
            print(f"loading lora...... ==> {lora_group[lora_id]['lora']}")
            pipe.load_lora_weights(lora_group[lora_id]['lora'])
            if  'TILoRA' in lora_id and not haved_load_ti:
                print(f"loading ti...... ==> {getattr(LoRA, f'{lora_id}_TI').value}")
                state_dict = load_file(getattr(LoRA, f"{lora_id}_TI").value)
                pipe.load_textual_inversion(state_dict["clip_g"], token="seekoo_ti", text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
                pipe.load_textual_inversion(state_dict["clip_l"], token="seekoo_ti", text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
                haved_load_ti = True
        else:
            pipe.unload_lora_weights()
        lora_state = lora_id

    # Determine whether to use Ipadapter
    return any(ip_model.units_enable)


def param2input(param: dict, weights_dict):
    global lora_state
    global lora_group
    # public param
    resize_mode = param.pop('resize_mode')
    cn_weights = param.pop('cn_weights')
    start_control_step = param.pop('start_control_step')
    end_control_step = param.pop('end_control_step')
    cache_path = param.pop('cache_path')
    # cache_path = lora_group[lora_state]['cache'] if lora_state is not None and lora_state in lora_group.keys() else cache_path

    if 'multi_ip_scale' in param.keys():     # single image reference
        # Multi reference mode param
        structure_image = param.pop('structure_image')
        color_image = param.pop('color_image')
        structure_scale = param.pop('structure_scale')
        color_scale = param.pop('color_scale')

        is_pro = param.pop('is_pro')

        image_paths = param.pop('image_paths')
        multi_ip_scale = param.pop('multi_ip_scale')
        multi_add_mode = param.pop('multi_add_mode')

        if structure_image is not None and color_image is not None:
            images = [structure_image, color_image]
            ip_scale = [structure_scale, color_scale]
            add_mode = 'multi-pro' if is_pro else 'multi-stable'
            feature_mode ='avg_feature'
        elif image_paths is not None or cache_path != '':
            if image_paths is not None:
                image_paths = [path.name for path in image_paths]
                images = [Image.open(path).convert("RGB") for path in image_paths]
            else:
                images = [None]
            ip_scale = [multi_ip_scale]
            add_mode = multi_add_mode
            feature_mode = 'avg_embeds'
        else:
            print('Error: Multi Reference image input is error')
            return None
    else:
        # single reference mode param
        style_image = param.pop('style_image')
        ip_scale = [param.pop('ip_scale')]
        add_mode = param.pop('add_mode')

        if not isinstance(style_image, Image.Image):
            style_image = Image.fromarray(style_image)
        images = [style_image]
        feature_mode = 'simple'

    if None not in images:
        if resize_mode == ImageOperation.JUST_RESIZE.value:
            images = [ImageOperation.just_resize(img) for img in images]
        elif resize_mode == ImageOperation.CROP_RESIZE.value:
            images = [ImageOperation.crop_and_resize(img) for img in images]
        else:
            print("Only support two resize mode :   ==> 1).'just resize'; 2).'resize&crop'")
    
    assert param=={}
    
    return {
        'pil_images': images,
        'ip_scale': ip_scale,
        'feature_mode': feature_mode,
        'weights_enable': weights_dict[add_mode],
        'cn_weights': cn_weights,
        'denoise_control': {
            's_step': start_control_step,
            'e_step': end_control_step,
        },
        'cache_paths': cache_path,
    }


def data_prepare(param_dict):
    global ip_model
    global lora_state

    # Check whether all IPadapters are disabled
    if not check_pipe(param_dict):    
        print('** Error: There are not ipadater is enable')  
        return None

    # get unit param
    base_param = param_dict.pop('base')
    xy_param = param_dict.pop('xy')
    controlnet_param = param_dict.pop('controlnet')
    lora_param = param_dict.pop('lora')
    ip_units_param = param_dict

    # 1. Prepare 
    # (1) prompt & negative prompt
    prompt_list = base_param.pop('prompt').split(';')
    negative_prompt = base_param.pop('negative_prompt')
    negative_prompt = '' if negative_prompt is None else negative_prompt
    # (2) weights enable
    layer_num = 70 if 'sdxl' in args.ip_ckpt else 16
    print(f'\033[91m Layer Num: >>{layer_num}<< \033[0m')
    weights = [0]*layer_num
    weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6 = \
        weights[:], weights[:], weights[:], weights[:], weights[:], weights[:], weights[:]
    # layer-wise for single reference image
    weights_0[24+10: 24+10+36] = [1]*36
    weights_1[24+10+10: 24+10+10 +10] = [1]*10
    weights_2[24+10+10: 24+10+10 +5] = [1]*5
    weights_3[24+10+10+5: 24+10+10+5 +5] = [1]*5
    assert len(weights_0)==len(weights_1)==len(weights_2)==len(weights_3)==layer_num
    # layer-wise for multi reference image
    weights_4[:24+10+10+5] = [1]*(24+10+10+5)
    weights_5[24+10+10+5: 24+10+10+5+5] = [1]*5
    weights_6[24+10+10: 24+10+10+10] = [1]*10
    weights_dict = {
        'full': [[1]*layer_num],
        'upblocks-only': [weights_0],
        '10layers': [weights_1],
        'front5layers': [weights_2],
        'last5layers': [weights_3],

        'multi-stable': [weights_4, weights_5],
        'multi-pro': [weights_4, weights_6],
    }
    # (3) xy fun 
    xy_mode = xy_param.pop('mode')
    preset = xy_param.pop('preset')
    x_type = xy_param.pop('x_type')
    x_value = xy_param.pop('x_value')
    y_type = xy_param.pop('y_type')
    y_value = xy_param.pop('y_value')
    # (4) controlnet param
    is_control = controlnet_param.pop('is_control')
    control_map = None
    # (5) fixed the naming of the 'add mode'
    for weights_key in weights_dict.keys():
        for _, unit_param in ip_units_param.items():
            if weights_key in unit_param['add_mode']:
                unit_param['add_mode'] = weights_key
            if weights_key in unit_param['multi_add_mode']:
                unit_param['multi_add_mode'] = weights_key

        for i, preset_mode in enumerate(preset):       # for xy preset mode
            if weights_key in preset_mode:
                preset[i] = weights_key
                break
    
    # 2. Convert data for each Ip Unit
    ip_units_input = {}
    for unit_id in ip_units_param.keys():
        ip_unit_param = ip_units_param[unit_id]
        single_enable = ip_unit_param.pop('single_enable')

        public_param = {
            'resize_mode': ip_unit_param.pop('resize_mode'),
            'cn_weights': ip_unit_param.pop('cn_weights'),
            'start_control_step': ip_unit_param.pop('start_control_step'),
            'end_control_step': ip_unit_param.pop('end_control_step'),
            'cache_path': ip_unit_param.pop('cache_path'),
        }

        if single_enable:
            param = {
                'style_image': ip_unit_param.pop('style_image'),
                'ip_scale': ip_unit_param.pop('ip_scale'),
                'add_mode': ip_unit_param.pop('add_mode'),
            }
        else:       
            param = {
                'structure_image': ip_unit_param.pop('structure_image'),
                'color_image': ip_unit_param.pop('color_image'),
                'is_pro': ip_unit_param.pop('is_pro'),
                'structure_scale': ip_unit_param.pop('structure_scale'),
                'color_scale': ip_unit_param.pop('color_scale'),
                'image_paths': ip_unit_param.pop('image_paths'),
                'multi_ip_scale': ip_unit_param.pop('multi_ip_scale'),
                'multi_add_mode': ip_unit_param.pop('multi_add_mode'),
            }
        
        param.update(public_param)
        ip_param = param2input(param, weights_dict)
        for k in list(ip_param.keys()):
            ip_units_input.setdefault(k, []).append(ip_param.pop(k))
        assert ip_param == {}
        
    # prepare for CrossAttention
    ip_units_input['cross_attention_kwargs'] = {
        'weights_enable': ip_units_input.pop('weights_enable'),
        'cn_weights': ip_units_input.pop('cn_weights'),
        'units_num_token': [ip_model.units_num_token[i] for i, enable in enumerate(ip_model.units_enable) if enable is True],
        'scale': lora_param.pop('lora_scale')
        }
    
    # 3. Process
    outputs = []
    result = None
    # Multiple prompt running diagram logic, different prompt use ';' partitions
    for prompt in prompt_list:
        if xy_mode == 'None':
            if lora_state is not None and 'TILoRA' in lora_state and ip_units_input['cross_attention_kwargs']['scale'] > 0:       # for TI lora
                prompt = 'seekoo_ti' + prompt
            if is_control:
                result_, control_map = controlnet_inference(prompt, negative_prompt, **{**base_param, **ip_units_input, **controlnet_param})   
            else:
                result_ = inference(prompt, negative_prompt, **{**base_param, **ip_units_input})
                 
        elif xy_mode == 'Preset':
            if 'TILoRA' in lora_state and ip_units_input['cross_attention_kwargs']['scale'] > 0:       # for TI lora
                prompt = 'seekoo_ti' + prompt
            base_param.pop('batch_size')
            ip_units_input.pop('ip_scale')
            result_ = xy_inference(prompt, negative_prompt, weights_dict, preset, **{**base_param, **ip_units_input},)
        
        elif xy_mode == 'Self-Defined':
            def _get_xy_values(xy_value: str):
                """ x value / y value must conform to the a+b(+c) format"""
                try:
                    a, b , c = xy_value.split('+')
                    b, c = float(b.split('(')[0]), float(c.split(')')[0])
                    ip_scales = np.arange(float(a), b+c, c).tolist()
                    return ip_scales
                except Exception as e:
                    print(f"Error {e}: X/Y Value is not illegal!! Please check it again")
                    return []
            
            def _get_var(x_type):
                if x_type != 'None' and 'Unit' in x_type:
                    x_id = int(x_type.split('-')[0].split('Unit')[1])
                    x_var = x_type.split('-')[1]
                elif x_type == 'None':
                    x_id, x_var = None, None
                else :
                    x_id, x_var = None, x_type
                return x_id, x_var
            
            x_id, x_var = _get_var(x_type)
            x_value = _get_xy_values(x_value)
            
            y_id, y_var = _get_var(y_type)
            y_value = _get_xy_values(y_value)
            
            xy_group = [(var, value, unit_id) for var, value, unit_id in zip([x_var, y_var], [x_value, y_value], [x_id, y_id]) \
                        if var is not None]

            # init
            base_param['batch_size'] = 1
            var_num = len(xy_group)
            process_index = 0
            result_ = None 
            img_hconcat = None
            img_vconcat = None

            def xy_selfdefined_process(x_tuple):
                nonlocal process_index
                nonlocal img_hconcat
                nonlocal img_vconcat
                nonlocal control_map
                nonlocal result_
                nonlocal prompt
                process_index += 1

                var, value, unit_id = x_tuple
                img_vconcat = None
                for value_ in value:
                    if var == 'lora_scale':
                        ip_units_input['cross_attention_kwargs']['scale'] = value_
                    elif var == 'control_weights':
                        controlnet_param['control_weights'] = value_
                    elif var in ['ip_scale', 'multi_ip_scale']:
                        ip_units_input['ip_scale'][unit_id][0] = value_
                    elif var == 'structure_scale':
                        ip_units_input['ip_scale'][unit_id][0] = value_
                    elif var == 'color_scale':
                        ip_units_input['ip_scale'][unit_id][1] = value_
                    else:
                        print("**current X/Y type is not support !!")
                        break

                    assert var_num <= 2, "**Error x/y var num must <= 2"

                    prompt_ = 'seekoo_ti' + prompt if lora_state is not None and 'TILoRA' in lora_state and ip_units_input['cross_attention_kwargs']['scale'] > 0 \
                        else prompt

                    if var_num == 1:        # only x
                        if is_control:
                            output, control_map = controlnet_inference(prompt_, negative_prompt, **{**base_param, **ip_units_input, **controlnet_param})   
                        else:
                            output = inference(prompt_, negative_prompt, **{**base_param, **ip_units_input})
                        result_ = cv2.hconcat([result_, output]) if result_ is not None else output

                    else:      # x-y        # var_num=2
                        if process_index == var_num:
                            if is_control:
                                output, control_map = controlnet_inference(prompt_, negative_prompt, **{**base_param, **ip_units_input, **controlnet_param})   
                            else:
                                output = inference(prompt_, negative_prompt, **{**base_param, **ip_units_input})
                            img_vconcat = cv2.vconcat([img_vconcat, output]) if img_vconcat is not None else output
                        
                        if process_index == var_num-1:
                            xy_selfdefined_process(xy_group[process_index])
                            result_ = cv2.hconcat([result_, img_vconcat]) if result_ is not None else img_vconcat

                process_index -= 1
            
            xy_selfdefined_process(xy_group[0])
                
        result = cv2.vconcat([result, result_]) if result is not None else result_
        outputs.append(Image.fromarray(result))

    # Only the pre-processed effect of the input image will be printed for a prompt
    for unit_input_imgs in ip_units_input['pil_images']:
        # only ip units that use a single reference will feed the input img into the outputs cache
        outputs += unit_input_imgs \
             if (len(unit_input_imgs)==1 and len(prompt_list) == 1) and (xy_mode == 'None') and (None not in unit_input_imgs) else []
    if is_control:
        outputs.append(control_map)

    return outputs 


def inference(prompt, negative_prompt, **kwargs):
    batch_size = kwargs.pop('batch_size')
    seed = kwargs.pop('seed')

    img_hconcat = None
    for i in range(batch_size):
        output = ip_model.generate(
            prompt, 
            negative_prompt, 
            num_inference_steps=20, 
            num_samples=1,
            seed=seed+i,
            **kwargs,
            )[0]
        img_hconcat = cv2.hconcat([img_hconcat, np.array(output)]) if img_hconcat is not None else np.array(output)
    return img_hconcat


def controlnet_inference(prompt, negative_prompt, **kwargs):
    control_input = kwargs.pop('control_input')
    control_type = kwargs.pop('control_type')
    control_resize_mode = kwargs.pop('resize_mode')
    control_weights = kwargs.pop('control_weights')
    canny_low = kwargs.pop('canny_low')
    canny_high = kwargs.pop('canny_high')
    batch_size = kwargs.pop('batch_size')
    seed = kwargs.pop('seed')
    control_map = None

    assert control_input is not None, '** Error: control input is None'

    img_hconcat = None
    for i in range(batch_size):
        # w, h = control_input.size
        if control_resize_mode == ImageOperation.JUST_RESIZE.value:
            control_input = ImageOperation.just_resize(control_input)
        elif control_resize_mode == ImageOperation.CROP_RESIZE.value:
            control_input = ImageOperation.crop_and_resize(control_input)
        else:
            print("Only support two resize mode :   ==> 1).'just resize'; 2).'resize&crop'")
        control_map = control_preprocess(control_input, control_type, canny_low, canny_high)
        output = ip_model.generate(
            prompt, 
            negative_prompt, 
            num_inference_steps=20, 
            num_samples=1, 
            seed=seed+i,
            image=control_map,
            controlnet_conditioning_scale=float(control_weights),
            **kwargs,
            )[0]
        img_hconcat = cv2.hconcat([img_hconcat, np.array(output)]) if img_hconcat is not None else np.array(output)
    
    return img_hconcat, control_map


def xy_inference( prompt, negative_prompt, weights_dict, preset, pil_images, height, width, seed, 
                 feature_mode, cross_attention_kwargs, **kwargs):
    global ip_model

    def _inference(scales_dict):
        result = None
        for mode, scale in scales_dict.items():
            weights = weights_dict[mode.split('-')[1]]
            cross_attention_kwargs['weights_enable'] = [weights]
            out = ip_model.generate(
                            prompt=prompt,
                            negative_prompt='' if negative_prompt is None else negative_prompt,
                            pil_images=pil_images, 
                            num_samples=1, 
                            num_inference_steps=20, 
                            height=height,
                            width=width,
                            seed=seed, 
                            guidance_scale=7.5,
                            scale=scale,
                            cross_attention_kwargs=cross_attention_kwargs,
                            feature_mode=feature_mode,
                            **kwargs
                            )[0]
            out = np.array(out)
            result = cv2.hconcat([result, out]) if result is not None else out
        return result
    if '8 preset setting' in preset:
        scales_dict = {
        'mode_0-last5layers': 1.6,
        'mode_1-10layers': 0.6,
        'mode_2-front5layers': 1.2,
        'mode_3-10layers': 1.2,
        'mode_4-full': 0.3,
        'mode_5-full': 0.35,
        'mode_6-full': 0.4,
        'mode_7-full': 0.5,
        }
        result = _inference(scales_dict)
        
    elif '6 preset setting' in preset:
        scales_dict = {
        'mode_0-10layers': 0.7,
        'mode_1-10layers': 1.0,
        'mode_2-10layers': 1.2,
        'mode_3-full': 0.35,
        'mode_4-full': 0.4,
        'mode_5-full': 0.45,
        }
        result = _inference(scales_dict)
    
    else:
        scales_dict = {
            'full': np.arange(0.1, 0.5+0.05, 0.05).tolist(),
            'upblocks-only': np.arange(0.1, 0.5+0.05, 0.05).tolist(),
            '10layers': np.arange(0.4, 1.2+0.1, 0.1).tolist(),
            'front5layers': np.arange(0.4, 1.2+0.1, 0.1).tolist(),
            'last5layers': np.arange(0.8, 1.6+0.1, 0.1).tolist(),
        }

        result = None
        for mode in preset:
            h_concat = None
            cross_attention_kwargs['weights_enable'] = weights_dict[mode]
            for scale in scales_dict[mode]:
                out = ip_model.generate(
                        prompt=prompt,
                        negative_prompt='' if negative_prompt is None else negative_prompt,
                        pil_images=pil_images, 
                        num_samples=1, 
                        num_inference_steps=20, 
                        height=height,
                        width=width,
                        seed=seed, 
                        guidance_scale=7.5,
                        scale=scale,
                        cross_attention_kwargs=cross_attention_kwargs,
                        feature_mode=feature_mode,
                    )[0]
                out = np.array(out)
                h_concat = cv2.hconcat([h_concat, out]) if h_concat is not None else out

            result = cv2.vconcat([result, h_concat]) if result is not None else h_concat

    return Image.fromarray(result)
    

def control_preprocess(image, control_type, canny_low, canny_high):
    if control_type is None:
        return None
    elif control_type == 'Depths':
        return get_depths(image)
    elif control_type == 'Canny':
        return get_canny(image, canny_low, canny_high)
    elif control_type == 'Lineart':
        return get_lineart(image)
    else:
        print('Something went wrong')
        return None


def main(port=10050):
    with gr.Blocks() as demo:
        # gr.Markdown("# IPAdapter Style Transfer")
        gr.Markdown("<div align='center' ><font size='20'>IPAdapter Style Transfer</font></div>")
        units_set = set() 

        # 1. Basic Unit
        with gr.Column(variant='compact'):
            prompt = gr.Textbox(label="prompt", elem_id='base-prompt')
            negative_prompt = gr.Textbox(label="negative prompt", elem_id='base-negative_prompt')
            button = gr.Button("Submit")
        with gr.Accordion('Other Param', open=False):
            with gr.Row(variant='compact'):
                height = gr.Number(label="heigh", value=1024, precision=0, elem_id='base-height')
                width = gr.Number(label="width", value=1024, precision=0, elem_id='base-width')
                seed = gr.Number(label="seed", value=42, precision=0, elem_id='base-seed')
                batch_size = gr.Number(label="batch size", value=1, precision=0, elem_id='base-batch_size')
            trick = gr.CheckboxGroup(
                [
                OtherTrick.IP_KV_NORM,
                OtherTrick.UNCOND_IMG_EMBEDS_CACHE,
                OtherTrick.GRAY_UNCOND_IMG_EMBEDS,
                ],
                label='Other Trick',
                elem_id='base-trick',
            )
            # midjourney_trick = gr.Checkbox(elem_id=f'base-midjourney_trick', label='Midjourney trick enable')
            # uncond_img_embeds = gr.Checkbox(elem_id=f'base-uncond_img_embeds', label='Uncond image embeds')
        units_set = units_set.union({prompt, negative_prompt, height, width, seed, batch_size, trick})
        
        # 2. Main Unit
        with gr.Row(variant='compact'):
            # IPAdapter Unit
            with gr.Tabs():
                for i in range(max_unit_count):
                    with gr.Tab(f"Unit{i}"):
                        unit = IPAdapterUi(model_dir=model_dir, tabname=f"Unit{i}")
                        model_id, input_set = unit.unit_group

                        def update_unit(model_id): 
                            for grad_com, model_id in model_id.items():                       
                                unit_id = int(grad_com.elem_id.split('-')[0].split('Unit')[1])
                                model_id = model_id

                            if model_id == 'None':
                                ip_model.ip_units[unit_id]=None
                                return model_id
                            
                            ip_ckpt = os.path.join(model_dir, model_id)
                            image_encoder_path = args.vit_h if 'vit-h' in model_id else args.vit_g
                            num_tokens = 16 if 'plus' in ip_ckpt else 4

                            ip_model.update_unit(unit_id, param_dict={
                                'image_encoder_path': image_encoder_path,
                                'ip_ckpt': ip_ckpt,
                                'num_tokens': num_tokens,
                                })
                            print(f'++ Unit{unit_id} finish update!')

                            return model_id

                        model_id.change(fn=update_unit, inputs={model_id}, outputs=model_id)
                    units_set = units_set.union(input_set)

            # Output
            # output = gr.Image(label='output', type='pil')   
            output = gr.Gallery(label='output', columns=4, height="auto", preview=True) 
        
        # 3. LoRA
        with gr.Accordion("LoRA", open=False):
            # def lora_update():
            #     result = [None]
            #     lora_ids = LoRA.__members__.keys()
            #     for lora_id in lora_ids:
            #         if 'IPLoRA' in lora_id:
            #             result.append(lora_id) if 'Cache' not in lora_id else None
            #         else:
            #             result.append(lora_id)
            #     return gr.Radio.update(choices=result)
            def lora_update():
                global lora_group
                lora_dirs = [member.value for member in LoRA.__members__.values()]
                for lora_dir in lora_dirs:
                    safetensors_group = [file_name for file_name in os.listdir(lora_dir) if file_name.endswith('.safetensors')]
                    pt_group = [file_name for file_name in os.listdir(lora_dir) if file_name.endswith('.pt')]
                    # for iplora-face_plus
                    if 'iplora-face_plus' in lora_dir:
                        for lora_file_name in safetensors_group:
                            if '-epoch-' in lora_file_name:
                                lora_id = lora_file_name.split('-epoch-')[0]
                            else:
                                print('**Error: An invalid lora file appears!!!')
                                return [None]
                            lora_name = lora_file_name.split('.')[0]
                            lora_group.setdefault(lora_name, {})
                            cache_match = False
                            for cache_file_name in pt_group:
                                if '-ip_image_embeddings' in cache_file_name:
                                    cache_id = cache_file_name.split('-ip_image_embeddings')[0]
                                else:
                                    print('**Error: An invalid cache file appears!!!')
                                    return [None]
                                if lora_id==cache_id:
                                    lora_group[lora_name]['lora'] = os.path.join(lora_dir, lora_file_name)
                                    lora_group[lora_name]['cache'] = os.path.join(lora_dir, cache_file_name)
                                    cache_match = True
                                    break
                            if not cache_match:
                                print("**Error: not cahce file match ==> {lora_file_name}")
                                return [None]
                    # for iplora
                    elif '21Lora' in lora_dir or '20Lora' in lora_dir:
                        for lora_file_name in safetensors_group:
                            if '_SDXLIP_' in lora_file_name:
                                lora_id = lora_file_name.split('_SDXLIP_')[0].replace("_", "")
                            else:
                                print('**Error: An invalid lora file appears!!!')
                                return [None]
                            lora_group.setdefault(lora_id, {})
                            cache_match = False
                            for cache_file_name in pt_group:
                                if '_xl_1024' in cache_file_name:
                                    cache_id = cache_file_name.split('_xl_1024')[0].replace("_", "")
                                else:
                                    print('**Error: An invalid cache file appears!!!')
                                    return [None]
                                if lora_id==cache_id:
                                    lora_group[lora_id]['lora'] = os.path.join(lora_dir, lora_file_name)
                                    lora_group[lora_id]['cache'] = os.path.join(lora_dir, cache_file_name)
                                    pt_group.remove(cache_file_name)
                                    cache_match = True
                                    break
                            if not cache_match:
                                print(f"**Error: not cahce file match ==> {lora_file_name}")
                                return [None]
                        if len(pt_group) != 0:
                            print(f">>{len(pt_group)}<< pt files don't match ==> {pt_group}")
                    
                    else:
                        print(f"**Error:  An invalid lora dir appears ==> {lora_dir}")
                        return [None]


                return [None] + list(lora_group.keys())

            with gr.Row():
                lora_refresh_buttn = gr.Button(
                    value=UiSymbol.refresh_symbol,
                    scale=1, size='sm', min_width=1, variant='tool'
                )
                lora_id = gr.Radio(
                    # choices = [None]+[m for m in LoRA.__members__.keys() if not('IPLoRA' in m and 'Cache' in m and m != 'lora_dir')], 
                    choices=lora_update(),
                    show_label=False, 
                    value=None,
                    elem_id='lora-lora_id',
                    scale=20,
                ) 
            lora_scale = gr.Slider(minimum=0.0, maximum=1.5, step=0.1, label="lora scale", value=1.0, elem_id=f"lora-lora_scale")
            cache_path = gr.Textbox(visible=False)
            def auto_update_cache(lora_id):
                # if lora_id is not None and 'IPLoRA' in lora_id:
                #     # cache_path = getattr(LoRA, f"{lora_id}_Cache").value if 'IPLoRA' in lora_id else ''
                #     cache_path = lora_group[lora_id]['cache']
                #     return gr.Textbox.update(visible=True, value=cache_path)
                if lora_id is not None and lora_id in lora_group.keys():
                    # cache_path = getattr(LoRA, f"{lora_id}_Cache").value if 'IPLoRA' in lora_id else ''
                    cache_path = lora_group[lora_id]['cache']
                    return gr.Textbox.update(visible=True, value=cache_path)
                else:
                    return gr.Textbox.update(visible=False)
            lora_id.change(fn=auto_update_cache, inputs=[lora_id], outputs=[cache_path])
            lora_refresh_buttn.click(fn=lora_update, outputs=[lora_id])

        units_set = units_set.union({lora_id, lora_scale})

        # 4. ControlNet Unit
        with gr.Accordion("ControlNet", open=False):
            with gr.Row(variant='compact'):
                with gr.Column(variant='compact'):
                    control_input = gr.Image(label='control input', type='pil', elem_id='controlnet-control_input')
                    preprocessor_button = gr.Button("preview")
                control_output = gr.Image(label='preprocessor preview', type='pil')
            with gr.Column(variant='compact'):
                is_control =  gr.Checkbox(label="enable", elem_id='controlnet-is_control')
                control_type = gr.Radio(["Depths", "Canny", "Lineart"], label='Control Type', elem_id='controlnet-control_type')
                with gr.Row(visible=False) as canny_state:
                    canny_low = gr.Slider(label='Canny LoW Threshold', value=100, minimum=0, maximum=255, elem_id='controlnet-canny_low')
                    canny_high = gr.Slider(label='Canny High Threshold', value=200, minimum=0, maximum=255, elem_id='controlnet-canny_high')
                control_weights = gr.Slider(label='Control Weights', value=1.0, minimum=0, maximum=1.6, elem_id='controlnet-control_weights')
                def update_canny_states(ctrl_type):
                    return gr.Row.update(visible=True) if ctrl_type == 'Canny' else gr.Row.update(visible=True)
                control_type.change(fn=update_canny_states, inputs=control_type, outputs=canny_state)

                with gr.Accordion("Other Control Setting", open=False):
                    with gr.Row(variant='compact'):
                        resize_mode = gr.Radio(
                            choices=[ImageOperation.JUST_RESIZE, ImageOperation.CROP_RESIZE],
                            value=ImageOperation.CROP_RESIZE,
                            label="resize mode", 
                            elem_id=f'controlnet-resize_mode',
                                )
                        control_mode = gr.Radio(
                            choices=[ControlMode.BALANCED, ControlMode.PROMPT], 
                            label='Control Mode', 
                            value=ControlMode.BALANCED, 
                            elem_id='controlnet-control_mode',
                            )
        # control preprocess
        preprocessor_button.click(fn=control_preprocess, inputs=[control_input, control_type, canny_low, canny_high], outputs=control_output)
        units_set = units_set.union({control_input, is_control, control_type, control_weights, control_mode, resize_mode, canny_low, canny_high})
        
        # 5. X-Y Unit
        with gr.Accordion("X-Y", open=False):
            with gr.Column(variant='compact'):
                # is_xy =  gr.Checkbox(label="enable", elem_id='xy-is_xy')
                xy_mode = gr.Radio(
                    choices = ['None', 'Preset', 'Self-Defined'],
                    value='None',
                    show_label=False,
                    elem_id='xy-mode',
                )
                
                with gr.Group(visible=False) as preset:
                    xy_preset = gr.CheckboxGroup([
                        'Clasic Elegance    (last5layers)',
                        'Artistic Fusion    (front5layers)', 
                        'Expressive Vibe    (10layers)', 
                        'Bold Transformation    (upblocks-only)', 
                        'Fantasy World  (full)', 
                        '8 preset setting',
                        '6 preset setting'], 
                        label='mode', 
                        info="The x-y function only supports single reference image and multiple reference images of the same style.\n\
                            The x-y function can display the generated map results of different modes and scales at one time. \n \
                            However, it should be noted that the preset values of scale have been set according to experimental analysis. \n \
                            'full':0.1-0.5(+0.05); \
                            'upblocks-only':0.1-0.5(+0.05); \
                            '10layers':0.4-1.2(+0.1);\
                            'front5layers':0.4-1.2(+0.1);\
                            'last5layers':0.8-1.6(+0.1);",
                        elem_id='xy-preset',
                    )
                
                with gr.Group(visible=False) as self_defined:
                    with gr.Row(variant='compact'):
                            x_type = gr.Dropdown(choices=['None'], value='None', label='X-Type', elem_id='xy-x_type',scale=20)
                            x_refresh_buttn = gr.Button(
                                value=UiSymbol.refresh_symbol,
                                # tooltip=UiSymbol.tooltips[UiSymbol.refresh_symbol],
                                scale=1, size='sm', min_width=1, variant='tool'
                            )
                            x_value = gr.Textbox(label="X-Value", elem_id='xy-x_value', scale=20)
                    with gr.Row(variant='compact'):
                            y_type = gr.Dropdown(choices=['None'], value='None', label='Y-Type', elem_id='xy-y_type',scale=20)
                            y_refresh_buttn = gr.Button(
                                value=UiSymbol.refresh_symbol,
                                # tooltip=UiSymbol.tooltips[UiSymbol.refresh_symbol],
                                scale=1, size='sm', min_width=1, variant='tool'
                            )
                            y_value = gr.Textbox(label="Y-Value", elem_id='xy-y_value', scale=20)
                
                    # update visible state
                    def xy_mode_change(xy_mode):
                        if xy_mode == 'Preset':
                            return [gr.update(visible=True), gr.update(visible=False)]
                        elif xy_mode == 'Self-Defined':
                            return [gr.update(visible=False), gr.update(visible=True)]
                        else:
                            return [gr.update(visible=False), gr.update(visible=False)]
                    xy_mode.change(fn=xy_mode_change, inputs=xy_mode, outputs=[preset, self_defined])

                    # update type list
                    def update_type(units_set):
                        # choices = ['lora_scale', 'control_weights']
                        choices = ['None']
                        param_dict = {}

                        try:
                            unit_keys = list(units_set.keys())
                            for key in unit_keys:
                                unit_id, param_id = key.elem_id.split('-')

                                param_dict.setdefault(unit_id, {})
                                param_dict[unit_id][param_id] = units_set[key]
                                    
                        except ValueError as e:
                            print(f'** {e} ==> key:{key.elem_id} \t value:{units_set[key]}')
                        
                        param_dict.pop('base')
                        for param_key, param in param_dict.items():
                            if 'Unit' in param_key:
                                unit_single_enable = param['single_enable']
                                unit_multi_enable = param['multi_enable']
                                if (unit_single_enable and unit_multi_enable) or \
                                    (not unit_single_enable and not unit_multi_enable):
                                    continue
                                if unit_single_enable:      # single reference mode
                                    choices.append(f"{param_key}-ip_scale")
                                else:       # multi reference mode
                                    choices.append(f"{param_key}-structure_scale")
                                    choices.append(f"{param_key}-color_scale")
                                    choices.append(f"{param_key}-multi_ip_scale")
                            elif param_key == 'lora':
                                choices.append(f"lora_scale") if param['lora_id'] is not None else None
                            elif param_key == 'controlnet':
                                choices.append(f"control_weights") if param['is_control'] else None
                            else:
                                print(f"** current param key is not legal ==> {param_key}")
                                break

                        return gr.Dropdown.update(choices=choices)

                    x_refresh_buttn.click(fn=update_type, inputs=units_set, outputs=[x_type])
                    y_refresh_buttn.click(fn=update_type, inputs=units_set, outputs=[y_type])

        units_set = units_set.union({xy_mode, xy_preset, x_type, x_value, y_type, y_value})
            
        def unit_prepare(units_set):
            param_dict = {}
            try:
                for key, value in units_set.items():
                    unit_id, param_id = key.elem_id.split('-')
                    param_dict.setdefault(unit_id, {})
                    param_dict[unit_id][param_id] = value
            except ValueError as e:
                print(f'{e} ==> key:{key.elem_id} \t value:{value}')
            return data_prepare(param_dict)
 
        button.click(fn=unit_prepare, inputs=units_set, outputs=output)

    demo.queue(concurrency_count=2).launch(server_name="0.0.0.0", server_port=port, share=True)
    # demo.queue(concurrency_count=2).launch(server_port=port, share=True)


max_unit_count = 4
ip_model, noise_scheduler, unet, controlnet, pipe = None, None, None, None, None
pipe_state = None
lora_group = {}
lora_state = None
haved_load_ti = False

LAYER_NUM = 70
controlnet_mode = {
    'Depths': r'/mnt/nfs/file_server/public/mingjiahui/models/diffusers--controlnet-depth-sdxl-1.0/',
    'Canny': r'/mnt/nfs/file_server/public/mingjiahui/models/diffusers--controlnet-canny-sdxl-1.0/',
    'Lineart': r'/mnt/nfs/file_server/public/lipengxiang/sdxl_lineart',
}

model_dir = r'/mnt/nfs/file_server/public/mingjiahui/models/h94--IP-Adapter/h94--IP-Adapter/sdxl_models/'
save_dir = r'/mnt/nfs/file_server/public/mingjiahui/IPAdapter_UI/'
current_datatime = datetime.datetime.now()
year = current_datatime.year
month = current_datatime.month
day = current_datatime.day
save_dir = os.path.join(save_dir, f'{year}-{month}-{day}')
os.makedirs(save_dir, exist_ok=True)
args = set_parser()


if __name__ == '__main__':
    if not args.debug:
        print(r'loading model......')
        load_model(
        base_model_path=args.base_model_path,
        image_encoder_path=args.vit_h,
        ip_ckpt=args.ip_ckpt,
        unet_load=True,
        )
        pipe_state = 'base'
    main(port=args.port)        # 10050
