import argparse
from PIL import Image
import json
import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from ui_v2 import load_model, data_prepare

image_encoder_paths = {
    "ip-adapter-faceid_sdxl": "buffalo_l",
    "ip-adapter-plus-face_sdxl_vit-h": "/mnt/nfs/file_server/public/mingjiahui/models/h94--IP-Adapter/h94--IP-Adapter/models/image_encoder/",
}
model_dir = r'/mnt/nfs/file_server/public/mingjiahui/models/h94--IP-Adapter/h94--IP-Adapter/sdxl_models/'


def main(args):
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
        ip_ckpt.append(os.path.join(model_dir, v['model_id']))
        image_encoder_path.append(image_encoder_paths[v['model_id'].split('.')[0]])
    load_model(
        base_model_path=args.base_model_path,
        image_encoder_path=image_encoder_path,
        ip_ckpt=ip_ckpt,
        unet_load=True,
    )
    # prepare input image
    image = Image.open(args.input_image).convert("RGB")
    param_data['Unit0']['style_image'] = image
    param_data['Unit1']['style_image'] = image
    # prepare prompt
    suffix = os.path.basename(args.input_image).split('.')[1]
    txt_path = args.input_image.replace(suffix, 'txt')
    with open(txt_path, 'r')as f:
        prompts = f.readlines()
        assert len(prompts)==1, "txt file looks happening some error"
    param_data['base']['prompt'] = prompts[0]
    
    output = data_prepare(param_data, args)[0]
    save_path = r"/home/mingjiahui/projects/IpAdapter/IP-Adapter/my_script/ui_v2_experiment/output/debug.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output.save(save_path)
    print(f"result has save in ==> {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/")
    parser.add_argument("--json_path", type=str, 
                        default="/home/mingjiahui/projects/IpAdapter/IP-Adapter/my_script/ui_v2_experiment/json/experiment_for_faceid.json")
    parser.add_argument("--input_image", type=str, required=True)
    args = parser.parse_args()
    main(args)