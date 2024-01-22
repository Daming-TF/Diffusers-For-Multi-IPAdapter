import argparse
from PIL import Image
import cv2
import json
import os
import sys
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from diffusers import DDIMScheduler
import torch
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from ui_v2 import load_model, data_prepare
from my_script.util.util import FaceidAcquirer
from my_script.models.IPAdapterXL import StableDiffusionXLControlNetImg2ImgPipelineV1, StableDiffusionXLImg2ImgPipelineV1
from my_script.models.IPAdapter import UNet2DConditionModelV1 as UNet2DConditionModel
from diffusers import ControlNetModel


model_dir = r'/mnt/nfs/file_server/public/mingjiahui/models/h94--IP-Adapter/h94--IP-Adapter'
image_encoder_paths = {
    "ip-adapter_sdxl": f"{model_dir}/sdxl_models/image_encoder/",
    "ip-adapter-faceid_sdxl": None,
    "ip-adapter-plus_sdxl_vit-h": f"{model_dir}/models/image_encoder/",
    "ip-adapter-plus-face_sdxl_vit-h": f"{model_dir}/models/image_encoder/",
    "ip-adapter-faceid-plusv2_sdxl": f"{model_dir}/models/image_encoder/",
}
embeds_dir = "/mnt/nfs/file_server/public/lzx/repo/stable-diffusion-api/models/iplora/"
LoRAs = {
        '3dpixel2sd': f"{embeds_dir}/3DPixel2_SDXLIP_v0_embeddings.pt", 
        'papercutoutsd': f"{embeds_dir}/PaperCutout_SDXLIP_v0_embeddings.pt", 
        # '3dexaggeration_sd': f"{embeds_dir}/3DExaggeration_SDXLIP_v0_embeddings.pt", 
        # 'graffitisplash_sd': f"{embeds_dir}/GraffitiSplash_SDXLIP_v0_embeddings.pt", 
        # 'holographic_sd': f"{embeds_dir}/Holographic_SDXLIP_v0_embeddings.pt",
        'colorfulrhythmsd': f'{embeds_dir}/Colorful_Rhythm_xl_1024_ip_image_embeddings.pt',
        'joyfulcartoonsd': f'{embeds_dir}/Joyful_Cartoon_xl_1024_ip_image_embeddings.pt',
    }
# save_dir = r"./my_script/ui_v2_experiment/output/diff_pipe_comparison"
save_dir = r"/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/untreated_output/"
os.makedirs(save_dir, exist_ok=True)

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(1024)
])

app = FaceidAcquirer()


def pipe_process(image_paths, param_data, pipe_name, use_control):
    print(f" \
          base: \nbatch_size:{param_data['base']['batch_size']}\n\
          Unit0: \nmodel:{param_data['Unit0']['model_id']}\tip scale:{param_data['Unit0']['multi_ip_scale']}\n\
          Unit1: \nmodel:{param_data['Unit1']['model_id']}\tip scale:{param_data['Unit1']['ip_scale']}\n\
          Unit2: \nmodel:{param_data['Unit2']['model_id']}\tip scale:{param_data['Unit2']['ip_scale']}\n\
          ")
    for lora_id, embeds_path in LoRAs.items():
        for image_path in tqdm(image_paths):
            # (1)prepare save path
            dir_name = os.path.basename(os.path.dirname(image_path))
            save_dir_ = os.path.join(save_dir, dir_name, lora_id, pipe_name)
            os.makedirs(save_dir_, exist_ok=True)
            try:
                # (2)prepare input image
                image = transform(Image.open(image_path).convert("RGB"))
                for i in range(4):
                    if not param_data[f'Unit{i}']['single_enable']:
                        continue
                    if 'plus-face' in param_data[f'Unit{i}']['model_id']:
                        _, face_image = app.get_face_embeds(image)
                        param_data[f'Unit{i}']['style_image'] = Image.fromarray(face_image[:, :, ::-1])
                    else:
                        param_data[f'Unit{i}']['style_image'] = image
                if use_control:
                    param_data['controlnet']['control_input'] = image
                if param_data['base']['pipe_type'] == "img2img":
                    print("it's img2img")
                    param_data['base']['image'] = image

                # (3)prepare prompt
                suffix = os.path.basename(image_path).split('.')[1]
                txt_path = image_path.replace(suffix, 'txt')
                with open(txt_path, 'r')as f:
                    prompts = f.readlines()
                    assert len(prompts)==1, "txt file looks happening some error"
                param_data['base']['prompt'] = prompts[0]
                
                # (4)check exists result
                save_name = os.path.basename(image_path)
                save_path = os.path.join(save_dir_, save_name)
                if os.path.exists(save_path):
                    continue

                # (5)processing
                result = None
                # no style
                # input_param = deepcopy(param_data)
                # input_param['lora']['lora_id']=lora_id
                # input_param['lora']['lora_scale']=0
                # input_param['Unit0']['cache_path']=embeds_path
                # input_param['Unit0']['multi_ip_scale']=0
                # output = data_prepare(input_param, args)[0]
                # result = np.array(output)

                input_param = deepcopy(param_data)
                input_param['lora']['lora_id']=lora_id
                input_param['lora']['lora_scale']=1.0
                input_param['Unit0']['cache_path']=embeds_path
                input_param['Unit0']['multi_ip_scale']=0.8
                output = data_prepare(input_param, args)[0]
                result = cv2.vconcat([result, np.array(output)]) if result is not None \
                    else np.array(output)
                
            except Exception as e:
                print(f"********* {e}:==>{image_path} **************")
                continue
            Image.fromarray(result).save(save_path)
            print(f"result has save in ==> {save_path}")


def main(args):
    # init
    image_paths = [os.path.join(args.input_image_dir, name) for name in os.listdir(args.input_image_dir) \
                   if name.split('.')[1] in ['jpg', 'png', 'webp']]
    with open(args.json_path, 'r')as f:
        param_data = json.load(f)
    ip_ckpt = []
    image_encoder_path = []
    for k, v in param_data.items(): 
        if 'Unit' not in k:
            continue
        model_id = v['model_id']
        if model_id == 'None' or model_id == '':
            break
        ip_ckpt.append(os.path.join(model_dir, 'sdxl_models', model_id))
        image_encoder_path.append(image_encoder_paths[model_id.split('.')[0]])
    
    # load model
    # 1.pipe
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
    # pipe = StableDiffusionXLImg2ImgPipelineV1.from_pretrained(
    #     base_model_path,
    #     unet=unet,
    #     torch_dtype=torch.float16,
    #     scheduler=noise_scheduler,
    #     add_watermarker=False,
    # )
    # load_model(
    #     base_model_path=args.base_model_path,
    #     image_encoder_path=image_encoder_path,
    #     ip_ckpt=ip_ckpt,
    #     input_pipe=pipe,
    #     device=device,
    # )

    assert param_data['Unit0']['model_id'] == "ip-adapter_sdxl.bin"
    assert param_data['Unit1']['model_id'] == "ip-adapter-faceid-plusv2_sdxl.bin"
    assert param_data['Unit2']['model_id'] == "ip-adapter-plus_sdxl_vit-h.bin"
    # # pipe1: only faceid plus v2(0.8)
    # # strength 0.6
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.8
    # input_param['Unit1']['face_id_lora']=0.8
    # input_param['base']['strength']=0.6
    # pipe_process(image_paths, input_param, \
    #              pipe_name="img2img--only_faceid_plusv2_0.8--strength_0.6", use_control=False)
    # # strength 1.0
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.8
    # input_param['Unit1']['face_id_lora']=0.8
    # input_param['base']['strength']=1.0
    # pipe_process(image_paths, input_param, \
    #              pipe_name="img2img--only_faceid_plusv2_0.8--strength_1.0", use_control=False)

    # # pipe2: faceid plus v2(0.8) + plus(0.3)
    # # strength 0.6
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.8
    # input_param['Unit1']['face_id_lora']=0.8
    # input_param['Unit2']['ip_scale']=0.3
    # input_param['base']['strength']=0.6
    # pipe_process(image_paths, input_param, \
    #              pipe_name="img2img--faceid_plusV2_0.8--plus_0.3--strength_0.6", use_control=False)
    # # strength 1.0
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.8
    # input_param['Unit1']['face_id_lora']=0.8
    # input_param['Unit2']['ip_scale']=0.3
    # input_param['base']['strength']=1.0
    # pipe_process(image_paths, input_param, \
    #              pipe_name="img2img--faceid_plusV2_0.8--plus_0.3--strength_1.0", use_control=False)

    controlnet_model_path = r'/mnt/nfs/file_server/public/mingjiahui/models/diffusers--controlnet-canny-sdxl-1.0/'
    controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetImg2ImgPipelineV1.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        unet=unet,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        add_watermarker=False,
    )
    load_model(
        base_model_path=args.base_model_path,
        image_encoder_path=image_encoder_path,
        ip_ckpt=ip_ckpt,
        input_pipe=pipe,
        device=device,
    )
    # pipe3: faceid plus v2(0.8) + controlnet 0.2
    # strength 0.6
    input_param = deepcopy(param_data)
    input_param['Unit1']['ip_scale']=0.8
    input_param['Unit1']['face_id_lora']=0.8
    input_param['controlnet']['is_control']=True
    input_param['controlnet']['control_weights']=0.2
    input_param['base']['strength']=0.6
    pipe_process(image_paths, input_param, \
                 pipe_name="img2img--faceid_plusV2_0.8--controlnet0.2--strength_0.6", use_control=True)
    # strength 1.0
    input_param = deepcopy(param_data)
    input_param['Unit1']['ip_scale']=0.8
    input_param['Unit1']['face_id_lora']=0.8
    input_param['controlnet']['is_control']=True
    input_param['controlnet']['control_weights']=0.2
    input_param['base']['strength']=1.0
    pipe_process(image_paths, input_param, \
                 pipe_name="img2img--faceid_plusV2_0.8--controlnet0.2--strength_1.0", use_control=True)

    # pipe4: faceid plus v2(0.8) + controlnet 0.2 + plus0.3
    # strength 0.6
    input_param = deepcopy(param_data)
    input_param['Unit1']['ip_scale']=0.8
    input_param['Unit1']['face_id_lora']=0.8
    input_param['controlnet']['is_control']=True
    input_param['controlnet']['control_weights']=0.2
    input_param['Unit2']['ip_scale']=0.3
    input_param['base']['strength']=0.6
    pipe_process(image_paths, input_param, \
                 pipe_name="img2img--faceid_plusV2_0.8--plus_0.3--controlnet0.2--strength_0.6", use_control=True)
    # strength 1.0
    input_param = deepcopy(param_data)
    input_param['Unit1']['ip_scale']=0.8
    input_param['Unit1']['face_id_lora']=0.8
    input_param['controlnet']['is_control']=True
    input_param['controlnet']['control_weights']=0.2
    input_param['Unit2']['ip_scale']=0.3
    input_param['base']['strength']=1.0
    pipe_process(image_paths, input_param, \
                 pipe_name="img2img--faceid_plusV2_0.8--plus_0.3--controlnet0.2--strength_1.0", use_control=True)

    # pipe5: faceid plus v2(0.8) + controlnet 0.5
    # strength 0.6
    input_param = deepcopy(param_data)
    input_param['Unit1']['ip_scale']=0.8
    input_param['Unit1']['face_id_lora']=0.8
    input_param['controlnet']['is_control']=True
    input_param['controlnet']['control_weights']=0.5
    input_param['base']['strength']=0.6
    pipe_process(image_paths, input_param, \
                 pipe_name="img2img--faceid_plusV2_0.8--controlnet0.5--strength_0.6", use_control=True)
    # strength 1.0
    input_param = deepcopy(param_data)
    input_param['Unit1']['ip_scale']=0.8
    input_param['Unit1']['face_id_lora']=0.8
    input_param['controlnet']['is_control']=True
    input_param['controlnet']['control_weights']=0.5
    input_param['base']['strength']=1.0
    pipe_process(image_paths, input_param, \
                 pipe_name="img2img--faceid_plusV2_0.8--controlnet0.5--strength_1.0", use_control=True)

    # pipe6: faceid plus v2(0.8) + controlnet 0.2 + plus0.3
    # strength 0.6
    input_param = deepcopy(param_data)
    input_param['Unit1']['ip_scale']=0.8
    input_param['Unit1']['face_id_lora']=0.8
    input_param['controlnet']['is_control']=True
    input_param['controlnet']['control_weights']=0.5
    input_param['Unit2']['ip_scale']=0.3
    input_param['base']['strength']=0.6
    pipe_process(image_paths, input_param, \
                 pipe_name="img2img--faceid_plusV2_0.8--plus_0.3--controlnet0.5--strength_0.6", use_control=True)
    # strength 1.0
    input_param = deepcopy(param_data)
    input_param['Unit1']['ip_scale']=0.8
    input_param['Unit1']['face_id_lora']=0.8
    input_param['controlnet']['is_control']=True
    input_param['controlnet']['control_weights']=0.5
    input_param['Unit2']['ip_scale']=0.3
    input_param['base']['strength']=1.0
    pipe_process(image_paths, input_param, \
                 pipe_name="img2img--faceid_plusV2_0.8--plus_0.3--controlnet0.5--strength_1.0", use_control=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/")
    # parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--json_path", type=str, default=None, required=True)
    parser.add_argument("--input_image_dir", type=str, required=True, help="image dir")
    args = parser.parse_args()
    main(args)