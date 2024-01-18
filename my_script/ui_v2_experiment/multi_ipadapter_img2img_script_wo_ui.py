import argparse
from PIL import Image
import cv2
import json
import os
import sys
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, DDIMScheduler
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from ui_v2 import load_model, data_prepare, inference
from my_script.models.IPAdapterXL import StableDiffusionXLImg2ImgPipelineV1
from my_script.models.IPAdapter import UNet2DConditionModelV1 as UNet2DConditionModel


embeds_dir = "/mnt/nfs/file_server/public/lzx/repo/stable-diffusion-api/models/iplora/"
embeds = {
        '3dpixel2_sd': f"{embeds_dir}/3DPixel2_SDXLIP_v0_embeddings.pt", 
        'papercutout_sd': f"{embeds_dir}/PaperCutout_SDXLIP_v0_embeddings.pt", 
        '3dexaggeration_sd': f"{embeds_dir}/3DExaggeration_SDXLIP_v0_embeddings.pt", 
        'graffitisplash_sd': f"{embeds_dir}/GraffitiSplash_SDXLIP_v0_embeddings.pt", 
        # 'holographic_sd': f"{embeds_dir}/Holographic_SDXLIP_v0_embeddings.pt"
    }
lora_dir = "/mnt/nfs/file_server/public/lzx/repo/stable-diffusion-api/models/Lora/iplora/lora_official/"
lora = {
        '3dpixel2_sd': f"{lora_dir}/3DPixel2_SDXLIP_v0.safetensors", 
        'papercutout_sd': f"{lora_dir}/PaperCutout_SDXLIP_v0.safetensors", 
        '3dexaggeration_sd': f"{lora_dir}/3DExaggeration_SDXLIP_v0.safetensors", 
        'graffitisplash_sd': f"{lora_dir}/GraffitiSplash_SDXLIP_v0.safetensors", 
        # 'holographic_sd': f"{embeds_dir}/Holographic_SDXLIP_v0_embeddings.pt"
    }
model_dir = "/mnt/nfs/file_server/public/mingjiahui/models/h94--IP-Adapter/h94--IP-Adapter/"
model_dict = {
    'faceid':{
        'ip_ckpt':f"{model_dir}/sdxl_models/ip-adapter-faceid_sdxl.bin",
        'image_encoder_path':"buffalo_l",
        'lora':f"{model_dir}/sdxl_models/ip-adapter-faceid_sdxl_lora.safetensors"
    },
    'base':{
        'ip_ckpt':f"{model_dir}/sdxl_models/ip-adapter_sdxl.bin",
        'image_encoder_path':f"{model_dir}/sdxl_models/image_encoder",
    },
    'plus':{
        'ip_ckpt':f"{model_dir}/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
        'image_encoder_path':f"{model_dir}/models/image_encoder",
    },
    'faceplus':{
        'ip_ckpt':f"{model_dir}/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin",
        'image_encoder_path':f"{model_dir}/models/image_encoder",
    },
}
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(1024)
])
save_dir = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/my_script/ui_v2_experiment/output/diff_pipe_comparison_img2img"
os.makedirs(save_dir, exist_ok=True)

def main(args):
    # 1. init
    # image_paths = [os.path.join(args.input, name) for name in os.listdir(args.input) \
    #                if name.split('.')[1] in ['jpg', 'png', 'webp']] if os.path.isdir(args.input) else [args.input]
    image_paths = [
        "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data/suren6.jpg",
        "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data/wang1.jpg",
        "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data/wang2.jpg",
        "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data/wangbaoqiang.jpg",
    ]
    with open(args.json_path, 'r')as f:     # json file for api enhanced vision
        input_param = json.load(f)
    start, end, inerval = args.ip_scales.split('-')
    ip_scales = np.arange(float(start), float(end)+float(inerval), float(inerval))
    start, end, inerval = args.strengths.split('-')
    strengths = np.arange(float(start), float(end)+float(inerval), float(inerval))
    enable_unit_id = [0,1,2]
    print(f"ip_scales:{ip_scales}\tstrengths:{strengths}")
    input_param['ip_scale'][1] = [args.faceid_ip_scale]
    
    # 2. load sdxl
    base_model_path = r"/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/"       # "SG161222/RealVisXL_V3.0"
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
    unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder='unet',
        ).to(dtype=torch.float16)
    pipe = StableDiffusionXLImg2ImgPipelineV1.from_pretrained(
        base_model_path,
        unet=unet,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        add_watermarker=False,
    )

    # 3.load lora
    lora_list = []
    lora_weight = []
    faceid_lora = model_dict['faceid']['lora']
    pipe.load_lora_weights(
            os.path.dirname(faceid_lora),
            weight_name=os.path.basename(faceid_lora), 
            adapter_name='faceid'
        )
    lora_list.append('faceid')
    lora_weight.append(args.faceid_lora_weight)
    
    if args.iplora is not None:
        input_param['cache_paths'][0] = embeds[args.iplora]
        lora_path = lora[args.iplora]
        pipe.load_lora_weights(
            os.path.dirname(lora_path),
            weight_name=os.path.basename(lora_path), 
            adapter_name=args.iplora
        )
        lora_list.append(args.iplora)
        lora_weight.append(args.iplora_weight)
    pipe.set_adapters(lora_list, adapter_weights=lora_weight)

    # 4. load ipadapter
    load_model(
        base_model_path=args.base_model_path,
        image_encoder_path=[model_dict[name.split('-')[0]]['image_encoder_path'] for name in args.model_list],
        ip_ckpt=[model_dict[name.split('-')[0]]['ip_ckpt'] for name in args.model_list],
        device=device,
        input_pipe=pipe,
        # unet_load=True,
    )

    # 5. set Multi ipadapter —— [ipscale]
    for model_index, model_name in enumerate(args.model_list):
        _, ip_scale = model_name.split('-')
        input_param['ip_scale'][model_index][0] = ip_scale

    # 6. processing
    for image_path in tqdm(image_paths):
        image = transform(Image.open(image_path).convert("RGB"))
        input_param['pil_images'][1][0] = image
        input_param['pil_images'][2][0] = image
        input_param['image'] = image

        suffix = os.path.basename(image_path).split('.')[1]
        txt_path = image_path.replace(suffix, 'txt')
        with open(txt_path, 'r') as f:
            data = f.readlines()
            assert len(data)==1
            prompt = data[0]

        result = None
        for strength in strengths:
            hconcat = None
            for ip_scale in ip_scales:
                input_param['strength'] = strength
                input_param['ip_scale'][0] = [ip_scale]
                output = inference(prompt, negative_prompt='', enable_unit_id=enable_unit_id, **input_param) # return np.ndarray
                hconcat = cv2.hconcat([hconcat, output]) if hconcat is not None else output

            result = cv2.vconcat([result, hconcat]) if result is not None else hconcat
        
            save_path = os.path.join(args.output, os.path.basename(image_path))
            Image.fromarray(result).save(save_path)
            print(f"result has saved in {save_path}")
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/")
    parser.add_argument("--json_path", type=str, 
        default="/home/mingjiahui/projects/IpAdapter/IP-Adapter/my_script/ui_v2_experiment/json/api_enhanced_version/iplora_faceid_plus_3ipadapter.json")
    parser.add_argument("--input", type=str, default="/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data", help="input image path")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--iplora", type=str, default=None)
    parser.add_argument("--iplora_weight", type=float, default=0.8)
    parser.add_argument("--faceid_lora_weight", type=float, default=0.7)
    # parser.add_argument("--faceid_ip_scale", type=float, default=0.7)
    parser.add_argument("--ip_scales", type=str, default="0-1-0.25")
    parser.add_argument("--strengths", type=str, default="0.2-1.0-0.2")
    parser.add_argument("--model_list", type=str, nargs='+', required=True, \
                        help="Union['base', 'faceid', 'plus', 'face plus']")        # e.g ['base-0.8', 'faceid-0.7', 'face plus-0.2']
    args = parser.parse_args()
    pipe_id = '-'.join(args.model_list)
    print(f"pipe id:{pipe_id}")
    if args.output is None:
        data_name =args.input if os.path.isdir(args.input) else os.path.dirname(args.input)
        args.output = os.path.join(save_dir, data_name+ '_img2img_script_output', \
                                   f"faceid_scale_{args.faceid_ip_scale}", str(args.iplora), pipe_id) 
    os.makedirs(args.output, exist_ok=True)
    print(f"Save dir:\t{args.output}")
    main(args)