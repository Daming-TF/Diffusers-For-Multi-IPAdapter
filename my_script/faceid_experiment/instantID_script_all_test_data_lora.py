import cv2
import torch
import numpy as np
from PIL import Image
import os
import argparse

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from InstantID.pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps
from InstantID.infer import resize_img
from my_script.util.util import image_grid


lora_dir = "/mnt/nfs/file_server/public/lzx/repo/stable-diffusion-api/models/Lora/iplora/lora_official"
lora_dict = {
    '3dpixel2sd': rf"{lora_dir}/3DPixel2_SDXLIP_v0.safetensors",
    'papercutoutsd': rf"{lora_dir}/PaperCutout_SDXLIP_v0.safetensors",
    'colorfulrhythmsd': rf"{lora_dir}/ColorfulRhythm_SDXLIP_v0.safetensors",
    'joyfulcartoonsd': rf"{lora_dir}/JoyfulCartoon_SDXLIP_v0.safetensors",
    # civit
    'ink': "/mnt/nfs/file_server/public/mingjiahui/models/lora-civit/liujiyou-SDXL.safetensors"
}
trigger_dict = {
    'ink': 'Chinese ink painting, traditional media, liujiyou'
}


def main(args):
    image_paths = [os.path.join(args.image_dir, name) for name in os.listdir(args.image_dir) if name.split('.')[1] in ['jpg', 'png']]
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

    # base_model_path = '/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/'
    base_model_path="/mnt/nfs/file_server/public/mingjiahui/models/wangqixun--YamerMIX_v8/"
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.cuda()
    pipe.load_ip_adapter_instantid(face_adapter)
    trigger = ''
    if args.lora is not None:
        lora_path = lora_dict[args.lora]
        print(f"loadding lora ==> {args.lora}")
        pipe.load_lora_weights(
            os.path.dirname(lora_path),
            weight_name=os.path.basename(lora_path), 
            adapter_name=args.lora
        )
        pipe.set_adapters([args.lora], adapter_weights=[1.0])
        trigger = trigger_dict.get(args.lora, '')
        

    for image_path in image_paths:
        suffix = os.path.basename(image_path).split('.')[1]
        txt_path = image_path.replace(suffix, 'txt')
        with open(txt_path, 'r')as f:
            prompt = trigger + f.readlines()[0]

        face_image = Image.open(image_path).convert("RGB")
        face_image = resize_img(face_image)

        face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]   # only use the maximum face
        face_emb = face_info['embedding']
        face_kps = draw_kps(face_image, face_info['kps'])

        pipe.set_ip_adapter_scale(0.8)
        images = []
        for _ in range(args.batch):
            generator = torch.Generator('cuda').manual_seed(42+_)
            image = pipe(
                prompt=prompt,
                image_embeds=face_emb,
                image=face_kps,
                controlnet_conditioning_scale=0.8,
                num_inference_steps=30,
                guidance_scale=5,
                # num_images_per_prompt=4,
                generator=generator
            ).images[0]
            images.append(image)
        reslut = image_grid(images, 2, 2)
        lora_id = '' if args.lora is None else f"{args.lora}_"
        save_path = os.path.join(args.save_dir, lora_id+os.path.basename(image_path))
        reslut.save(save_path)
        print(f"result has saved in ==> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data")
    parser.add_argument("--save_dir", type=str, default="/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/InstantID/script_output")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lora", type=str, default=None)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    assert args.lora in list(lora_dict.keys())+[None], ValueError(f"{args.lora} lora is not support")
    main(args)
    