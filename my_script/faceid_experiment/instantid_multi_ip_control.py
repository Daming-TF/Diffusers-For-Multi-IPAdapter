import cv2
import torch
import numpy as np
from PIL import Image
import os
import argparse
from torchvision import transforms

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis

import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from InstantID.pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from InstantID.pipeline_stable_diffusion_xl_instantid_mjh import StableDiffusionXLInstantIDPipelineCostom
from InstantID.infer import resize_img
from InstantID.ip_adapter.attention_processor import region_control
from my_script.util.util import image_grid, create_attention_mask


if __name__ == "__main__":
    # 1. init
    source_dir = "/home/mingjiahui/projects/IpAdapter/IP-Adapter"
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark_input", type=str, default=None, help="image path")
    parser.add_argument("--faceid_input", type=str, default=f'{source_dir}/data/all_test_data/luxun.jpg', help="image path")
    parser.add_argument("--save_dir", type=str, default=f'{source_dir}/data/InstantID/multi_ip_control')
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--control_weight", type=float, default=0.8)
    parser.add_argument("--ip_scale", type=float, default=0.8)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"result will save in ==> {args.save_dir}")
    transform = transforms.Resize(1024)
    reszie_crop = transforms.Compose([transform, transforms.CenterCrop(1024)])
    # suffix = os.path.basename(args.faceid_input).split('.')[1]
    # txt_path = args.replace(suffix, 'txt')
    # with open(txt_path, 'r')as f:
    #     prompt = f.readlines()[0]
    # experiment setting
    args.landmark_input = f'{source_dir}/data/InstantID/multi_ip_control_input.png'
    prompt = "Two men were standing in the garden, one in a black suit, the other is in a red suit and holding flowers"
    # prompt = "1 British Shorthair cat"
    # +++++++++++++++++++++++++++++++++++++++

    # 2.get face info
    app = FaceAnalysis(name='/home/mingjiahui/.insightface/models/antelopev2/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    landmark_input = resize_img(Image.open(args.landmark_input))
    t_w, t_h = landmark_input.size
    landmark_input_info = app.get(cv2.cvtColor(np.array(landmark_input), cv2.COLOR_RGB2BGR))
    landmark_input_info_list = sorted(landmark_input_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-2:] # only use the maximum face
    landmark_input_info_list = sorted(landmark_input_info_list, key=lambda x:x['bbox'][0])      # left to right
    assert len(landmark_input_info_list)==2, ValueError("some error has happened")
    landmark_input_info_list = sorted(landmark_input_info_list, key=lambda x:x['bbox'][0])    # left to right
    # landmark = draw_kps(landmark_input, [landmark_input_info['kps'] for landmark_input_info in landmark_input_info_list])
    landmark = draw_kps(landmark_input, [landmark_input_info_list[0]['kps']])
    landmark.save("./data/InstantID/debug_kps.jpg")

    # landmark_emb = landmark_input_info['embedding']
    landmark_input = np.array(landmark_input)
    faceid_input = Image.open(args.faceid_input)
    faceid_input_info = app.get(cv2.cvtColor(np.array(faceid_input), cv2.COLOR_RGB2BGR))
    faceid_input_info = sorted(faceid_input_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    face_emb = faceid_input_info['embedding']

    # 3.get attn_mask
    # attn_mask = create_attention_mask((t_h,t_w), [face_info['bbox'] for face_info in landmark_input_info_list])
    attn_mask = create_attention_mask((t_h,t_w), [landmark_input_info_list[0]['bbox']])
    Image.fromarray(attn_mask*255).save(f"{source_dir}/data/InstantID/debug_attn_mask.jpg")
    attn_mask = torch.from_numpy(attn_mask)
    # region_control.prompt_image_conditioning.append({'region_mask':attn_mask})

    # 3. prepare model
    # Path to InstantID models
    source_dir = f'/mnt/nfs/file_server/public/mingjiahui/models'
    face_adapter = f'{source_dir}/InstantX--InstantID/ip-adapter.bin'
    controlnet_path = f'{source_dir}/InstantX--InstantID/ControlNetModel'
    # Load pipeline
    print("loading controlnet")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, 
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    # base_model_path = '/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/'
    base_model_path="/mnt/nfs/file_server/public/mingjiahui/models/wangqixun--YamerMIX_v8/"
    print(f"loading sdxl ==> {base_model_path}")
    pipe = StableDiffusionXLInstantIDPipelineCostom.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.cuda()
    print("loading ipadapter")
    pipe.load_ip_adapter_instantid(face_adapter)
    print("finish loading!")

    # 4.processing
    pipe.set_ip_adapter_scale(args.ip_scale)
    images = []
    for i in range(args.batch):
        generator = torch.Generator('cuda').manual_seed(42+i)
        image = pipe(
            generator=generator,
            prompt=prompt,
            # negative_prompt=n_prompt,
            controlnet_conditioning_scale=args.control_weight,
            num_inference_steps=30,
            guidance_scale=5,
            # faceid 
            image_embeds=face_emb, 
            control_image_embeds=face_emb, 
            image=landmark, 
            # attention_mask=attn_mask,
            # cross_attention_kwargs={'attention_mask':attn_mask}
        ).images[0]
        images.append(image)
    grid = image_grid(images, int(args.batch**0.5), int(args.batch**0.5))
    
    faceid_input = np.array(reszie_crop(faceid_input).resize((t_h, t_h)))
    grid = np.array(grid.resize((t_w, t_h)))
    print(grid.shape, landmark_input.shape, faceid_input.shape)
    result = cv2.hconcat([grid, landmark_input, faceid_input])
    save_name = os.path.basename(args.faceid_input) if args.save_name is None else args.save_name
    save_path = os.path.join(args.save_dir, save_name)
    Image.fromarray(result).save(save_path)
    print(f"result has saved in ==> {save_path}")