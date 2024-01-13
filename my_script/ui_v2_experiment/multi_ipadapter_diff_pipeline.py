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
from my_script.util.util import FaceidAcquirer


model_dir = r'/mnt/nfs/file_server/public/mingjiahui/models/h94--IP-Adapter/h94--IP-Adapter'
image_encoder_paths = {
    "ip-adapter_sdxl": f"{model_dir}/sdxl_models/image_encoder/",
    "ip-adapter-faceid_sdxl": "buffalo_l",
    "ip-adapter-plus_sdxl_vit-h": f"{model_dir}/models/image_encoder/",
    "ip-adapter-plus-face_sdxl_vit-h": f"{model_dir}/models/image_encoder/",
    
}
embeds_dir = "/mnt/nfs/file_server/public/lzx/repo/stable-diffusion-api/models/iplora/"
LoRAs = {
        '3dpixel2_sd': f"{embeds_dir}/3DPixel2_SDXLIP_v0_embeddings.pt", 
        'papercutout_sd': f"{embeds_dir}/PaperCutout_SDXLIP_v0_embeddings.pt", 
        '3dexaggeration_sd': f"{embeds_dir}/3DExaggeration_SDXLIP_v0_embeddings.pt", 
        'graffitisplash_sd': f"{embeds_dir}/GraffitiSplash_SDXLIP_v0_embeddings.pt", 
        # 'holographic_sd': f"{embeds_dir}/Holographic_SDXLIP_v0_embeddings.pt"
    }
save_dir = r"./my_script/ui_v2_experiment/output/diff_pipe_comparison"
os.makedirs(save_dir, exist_ok=True)

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(1024)
])

app = FaceidAcquirer()



def pipe_process(image_paths, param_data, pipe_name, use_control):
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
                # if os.path.exists(save_path):
                #     continue

                # (5)processing
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
                input_param['lora']['lora_scale']=0.8
                input_param['Unit0']['cache_path']=embeds_path
                input_param['Unit0']['multi_ip_scale']=0.8
                output = data_prepare(input_param, args)[0]
                result = cv2.vconcat([result, np.array(output)])
                
                Image.fromarray(result).save(save_path)
                print(f"result has save in ==> {save_path}")
            except Exception as e:
                print(f"********* {e}:==>{image_path} **************")
                continue


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
    # CUDA0
    # # pipe0: only faceid(0.6)
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.6
    # input_param['Unit1']['face_id_lora']=0.6
    # pipe_process(image_paths, input_param, \
    #              pipe_name="only_faceid_0.6", use_control=False)

    # # pipe1: only faceid(0.7)
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.7
    # input_param['Unit1']['face_id_lora']=0.7
    # pipe_process(image_paths, input_param, \
    #              pipe_name="only_faceid_0.7", use_control=False)

    # # pipe2: faceid(0.7) + plus(0.2)
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.7
    # input_param['Unit1']['face_id_lora']=0.7
    # input_param['Unit2']['ip_scale']=0.2
    # pipe_process(image_paths, input_param, \
    #              pipe_name="faceid_0.7--plus_0.2", use_control=False)

    # # pipe3: faceid(0.7) + plus(0.5)
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.7
    # input_param['Unit1']['face_id_lora']=0.7
    # input_param['Unit2']['ip_scale']=0.5
    # pipe_process(image_paths, input_param, \
    #              pipe_name="faceid_0.7--plus_0.5", use_control=False)

    # # pipe4: faceid(0.7) + face plus(0.2)
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.7
    # input_param['Unit1']['face_id_lora']=0.7
    # input_param['Unit3']['ip_scale']=0.2
    # pipe_process(image_paths, input_param, \
    #              pipe_name="faceid_0.7--facePlus_0.2", use_control=False)

    # CUDA1
    # # pipe5: faceid(0.7) + face plus(0.5)
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.7
    # input_param['Unit1']['face_id_lora']=0.7
    # input_param['Unit3']['ip_scale']=0.5
    # pipe_process(image_paths, input_param, \
    #              pipe_name="faceid_0.7--facePlus_0.5", use_control=False)

    # # pipe6: faceid(0.7) + controlnet(0.2)
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.7
    # input_param['Unit1']['face_id_lora']=0.7
    # input_param['controlnet']['is_control']=True
    # input_param['controlnet']['control_weights']=0.2
    # pipe_process(image_paths, input_param, \
    #              pipe_name="faceid_0.7--control_0.2", use_control=True)

    # # pipe7: faceid(0.7) + controlnet(0.5)
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.7
    # input_param['Unit1']['face_id_lora']=0.7
    # input_param['controlnet']['is_control']=True
    # input_param['controlnet']['control_weights']=0.5
    # pipe_process(image_paths, input_param, \
    #              pipe_name="faceid_0.7--control_0.5", use_control=True)

    # CUDA2
    # pipe8: faceid(0.7) + controlnet(0.2) + face plus(0.2)
    input_param = deepcopy(param_data)
    input_param['Unit1']['ip_scale']=0.7
    input_param['Unit1']['face_id_lora']=0.7
    input_param['Unit3']['ip_scale']=0.2
    input_param['controlnet']['is_control']=True
    input_param['controlnet']['control_weights']=0.2
    pipe_process(image_paths, input_param, \
                 pipe_name="faceid_0.7--control_0.2--facePlus_0.2", use_control=True)

    # # pipe9: faceid(0.7) + controlnet(0.2) + plus(0.2)
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.7
    # input_param['Unit1']['face_id_lora']=0.7
    # input_param['Unit2']['ip_scale']=0.2
    # input_param['controlnet']['is_control']=True
    # input_param['controlnet']['control_weights']=0.2
    # pipe_process(image_paths, input_param, \
    #              pipe_name="faceid_0.7--control_0.2--plus_0.2", use_control=True)

    # # pipe10: faceid(0.7) + controlnet(0.5) + face plus(0.2)
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.7
    # input_param['Unit1']['face_id_lora']=0.7
    # input_param['Unit3']['ip_scale']=0.2
    # input_param['controlnet']['is_control']=True
    # input_param['controlnet']['control_weights']=0.5
    # pipe_process(image_paths, input_param, \
    #              pipe_name="faceid_0.7--control_0.5--facePlus_0.2", use_control=True)

    # CUDA3
    # # pipe11: faceid(0.7) + controlnet(0.5) + plus(0.2)
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.7
    # input_param['Unit1']['face_id_lora']=0.7
    # input_param['Unit2']['ip_scale']=0.2
    # input_param['controlnet']['is_control']=True
    # input_param['controlnet']['control_weights']=0.5
    # pipe_process(image_paths, input_param, \
    #              pipe_name="faceid_0.7--control_0.5--plus_0.2", use_control=True)

    # # pipe12: faceid(0.7) + plus(0.2) + face plus(0.2)
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.7
    # input_param['Unit1']['face_id_lora']=0.7
    # input_param['Unit2']['ip_scale']=0.2
    # input_param['Unit3']['ip_scale']=0.2
    # pipe_process(image_paths, input_param, \
    #              pipe_name="faceid_0.7--plus_0.2--facePlus_0.2", use_control=False)

    # # pipe13: faceid(0.7) + plus(0.2) + face plus(0.2) + controlnet(0.2)
    # input_param = deepcopy(param_data)
    # input_param['Unit1']['ip_scale']=0.7
    # input_param['Unit1']['face_id_lora']=0.7
    # input_param['Unit2']['ip_scale']=0.2
    # input_param['Unit3']['ip_scale']=0.2
    # input_param['controlnet']['is_control']=True
    # input_param['controlnet']['control_weights']=0.2
    # pipe_process(image_paths, input_param, \
    #              pipe_name="faceid_0.7--plus_0.2--facePlus_0.2--controlnet_0.2", use_control=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/")
    parser.add_argument("--json_path", type=str, 
        default="./my_script/ui_v2_experiment/json/faceid_xl_and_iplora_script--only_faceid.json")
    parser.add_argument("--input_image_dir", type=str, required=True, help="image dir")
    args = parser.parse_args()
    main(args)