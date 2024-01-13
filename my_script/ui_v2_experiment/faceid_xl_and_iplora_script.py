import argparse
from PIL import Image
import cv2
import json
import os
import sys
import numpy as np
from copy import deepcopy
from tqdm import tqdm
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from ui_v2 import load_model, data_prepare


model_dir = r'/mnt/nfs/file_server/public/mingjiahui/models/h94--IP-Adapter/h94--IP-Adapter'
image_encoder_paths = {
    "ip-adapter-faceid_sdxl": "buffalo_l",
    "ip-adapter-plus-face_sdxl_vit-h": f"{model_dir}/models/image_encoder/",
    "ip-adapter_sdxl": f"{model_dir}/sdxl_models/image_encoder/",
}
embeds_dir = "/mnt/nfs/file_server/public/lzx/repo/stable-diffusion-api/models/iplora/"
LoRAs = {
        '3dpixel2_sd': f"{embeds_dir}/3DPixel2_SDXLIP_v0_embeddings.pt", 
        'papercutout_sd': f"{embeds_dir}/PaperCutout_SDXLIP_v0_embeddings.pt", 
        '3dexaggeration_sd': f"{embeds_dir}/3DExaggeration_SDXLIP_v0_embeddings.pt", 
        'graffitisplash_sd': f"{embeds_dir}/GraffitiSplash_SDXLIP_v0_embeddings.pt", 
        'holographic_sd': f"{embeds_dir}/Holographic_SDXLIP_v0_embeddings.pt"
    }
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(1024)
])


def main(args):
    # init
    image_paths = [os.path.join(args.input_image_dir, name) for name in os.listdir(args.input_image_dir) \
                   if name.split('.')[1] in ['jpg', 'png', 'webp']]
    
    # load model
    with open(args.json_path, 'r')as f:
        param_data = json.load(f)
    ip_ckpt = []
    image_encoder_path = []
    for k, v in param_data.items(): 
        if 'Unit' not in k:
            continue
        model_id = v['model_id']
        if model_id == 'None':
            break
        ip_ckpt.append(os.path.join(model_dir, 'sdxl_models', v['model_id']))
        image_encoder_path.append(image_encoder_paths[v['model_id'].split('.')[0]])
    load_model(
        base_model_path=args.base_model_path,
        image_encoder_path=image_encoder_path,
        ip_ckpt=ip_ckpt,
        unet_load=True,
    )

    for lora_id, embeds_path in LoRAs.items():
        for image_path in tqdm(image_paths):
            try:
                # prepare input image
                image = transform(Image.open(image_path).convert("RGB"))
                param_data['Unit1']['style_image'] = image
                # prepare prompt
                suffix = os.path.basename(image_path).split('.')[1]
                txt_path = image_path.replace(suffix, 'txt')
                with open(txt_path, 'r')as f:
                    prompts = f.readlines()
                    assert len(prompts)==1, "txt file looks happening some error"
                param_data['base']['prompt'] = prompts[0]
                # prepare save path
                dir_name = os.path.basename(os.path.dirname(image_path))
                save_dir = os.path.join(r"./my_script/ui_v2_experiment/output", dir_name)
                os.makedirs(save_dir, exist_ok=True)
                
                save_dir_ = os.path.join(save_dir, lora_id)
                os.makedirs(save_dir_, exist_ok=True)
                save_name = os.path.basename(image_path)
                save_path = os.path.join(save_dir_, save_name)

                if os.path.exists(save_path):
                    continue
                # processing
                result = None
                input_param = deepcopy(param_data)
                input_param['lora']['lora_id']=lora_id
                input_param['lora']['lora_scale']=0
                input_param['Unit0']['cache_path']=embeds_path
                input_param['Unit0']['multi_ip_scale']=0
                output = data_prepare(input_param, args)[0]
                result = np.array(output)

                input_param = deepcopy(param_data)
                input_param['lora']['lora_id']=lora_id
                input_param['lora']['lora_scale']=0.6
                input_param['Unit0']['cache_path']=embeds_path
                input_param['Unit0']['multi_ip_scale']=0.6
                output = data_prepare(input_param, args)[0]
                result = cv2.vconcat([result, np.array(output)])
                
                Image.fromarray(result).save(save_path)
                print(f"result has save in ==> {save_path}")
            except Exception as e:
                print(f"********* {image_path} **************")
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/")
    parser.add_argument("--json_path", type=str, 
        default="./my_script/ui_v2_experiment/json/faceid_xl_and_iplora_script--only_faceid.json")
    parser.add_argument("--input_image_dir", type=str, required=True, help="image dir")
    args = parser.parse_args()
    main(args)