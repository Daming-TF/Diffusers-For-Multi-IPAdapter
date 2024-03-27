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
sys.path.append(os.path.dirname(current_path))
from instantid.pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from instantid.pipeline_stable_diffusion_xl_instantid_mjh import StableDiffusionXLInstantIDPipelineCostom as StableDiffusionXLInstantIDPipeline
from instantid.infer import resize_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark_input", type=str, required=True, help="image path")
    parser.add_argument("--faceid_input", type=str, required=True, help="image path")
    parser.add_argument("--save_dir", type=str, default='./data/InstantID/')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, \
        f"kps-{os.path.basename(args.landmark_input).split('.')[0]}--faceid-{os.path.basename(args.faceid_input).split('.')[0]}.jpg")
    print(f"result will save in ==> {save_path}")
    app = FaceAnalysis(name='/home/mingjiahui/.insightface/models/antelopev2/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Path to InstantID models
    source_dir = f'/mnt/nfs/file_server/public/mingjiahui/models'
    face_adapter = f'{source_dir}/InstantX--InstantID/ip-adapter.bin'
    controlnet_path = f'{source_dir}/InstantX--InstantID/ControlNetModel'

    # Load pipeline
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, 
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    base_model_path = '/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/'
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.cuda()
    pipe.load_ip_adapter_instantid(face_adapter)

    prompt = "A person with a black suit in the garden"
    n_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"

    transform = transforms.Resize(1024)
    reszie_crop = transforms.Compose([transform, transforms.CenterCrop(1024)])

    # face_image = load_image("./examples/yann-lecun_resize.jpg")
    landmark_input = resize_img(Image.open(args.landmark_input))
    faceid_input = Image.open(args.faceid_input)
    t_w, t_h = landmark_input.size

    landmark_input_info = app.get(cv2.cvtColor(np.array(landmark_input), cv2.COLOR_RGB2BGR))
    landmark_input_info = sorted(landmark_input_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    landmark = draw_kps(landmark_input, landmark_input_info['kps'])
    landmark_emb = landmark_input_info['embedding']


    faceid_input_info = app.get(cv2.cvtColor(np.array(faceid_input), cv2.COLOR_RGB2BGR))
    faceid_input_info = sorted(faceid_input_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    face_emb = faceid_input_info['embedding']
    
    # face_kps.save('./data/InstantID/debug.jpg')

    pipe.set_ip_adapter_scale(0)
    generator = torch.Generator('cuda').manual_seed(42)
    image = pipe(
        prompt=prompt,
        negative_prompt=n_prompt,
        controlnet_conditioning_scale=0.8,
        num_inference_steps=30,
        guidance_scale=5,
        generator=generator,
        # faceid 
        image_embeds=face_emb, 
        control_image_embeds=face_emb, 
        image=landmark, 
    ).images[0]

    landmark_input = np.array(landmark_input)
    faceid_input = np.array(reszie_crop(faceid_input).resize((t_h, t_h)))
    image = np.array(image.resize((t_w, t_h)))
    print(image.shape, landmark_input.shape, faceid_input.shape)
    result = cv2.hconcat([image, landmark_input, faceid_input])
    Image.fromarray(result).save(save_path)
    print(f"result has saved in ==> {save_path}")