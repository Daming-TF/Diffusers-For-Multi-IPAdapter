import cv2
from PIL import Image
from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import argparse
import numpy as np
import cv2
import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from my_script.util.util import image_grid, FaceidAcquirer
from my_script.ipadapter_xl_faceid_plus import CostomIPAdapterFaceIDPlusXL
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlusXL


def main(args):
    # 1.init
    v2 = True
    source_dir = "/mnt/nfs/file_server/public/mingjiahui/models"
    base_model_path = "/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/"     # "SG161222/Realistic_Vision_V4.0_noVAE"
    # vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path =  f"{source_dir}/laion--CLIP-ViT-H-14-laion2B-s32B-b79K"      # "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    ip_ckpt = f"{source_dir}/h94--IP-Adapter/h94--IP-Adapter/sdxl_models/ip-adapter-faceid-plusv2_sdxl.bin"
    assert 'v2' in ip_ckpt, "sdxl only support plusv2, not supoort faceid plus"
    device = "cuda"

    app = FaceidAcquirer()
    image_paths = [os.path.join(args.input, name)for name in os.listdir(args.input) if name.endswith('.jpg')]
    print(image_paths)

    print(f" save path: {args.save_dir}")
    print(f"\033[91m {ip_ckpt} \033[0m")
    print(f"\033[91m Test data num: {len(image_paths)} \033[0m")

    # 2.load model
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    # vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        # vae=vae,
        add_watermarker=False,
    )
    # trigger = "Chinese ink painting, traditional media, liujiyou"
    # lora_path = "/mnt/nfs/file_server/public/mingjiahui/models/lora-civit/liujiyou-SDXL.safetensors"
    # pipe.load_lora_weights(
    #     os.path.dirname(lora_path),
    #     weight_name=os.path.basename(lora_path),
    #     adapter_name='style'
    # )
    # pipe.set_adapters(adapter_names=['style'], adapter_weights=[1.0])

    # 3.load ip-adapter
    ip_model = CostomIPAdapterFaceIDPlusXL(pipe, image_encoder_path, ip_ckpt, device)
    ip_model.set_lora_scale(args.faceid_lora_weight)


    for image_path in image_paths:
        suffix = os.path.basename(image_path).split('.')[1]
        txt_path = image_path.replace(suffix, 'txt')
        if os.path.exists(txt_path):
            with open(txt_path, 'r')as f:
                data = f.readlines()
                assert len(data)==1
            # prompt = trigger + data[0]
            prompt = data[0]
        else:
            ValueError(f"The image has not correspoind txt file ==> {image_path}")
        print(f"prompt:{prompt}")
        
        # prepare face id
        faceid_embeds, face_image = app.get_face_embeds(cv2.imread(image_path))

        images = ip_model.generate(
            prompt=prompt, 
            # negative_prompt=negative_prompt, 
            face_image=face_image, faceid_embeds=faceid_embeds, 
            shortcut=v2, s_scale=args.s_scale, 
            scale=args.ip_scale,
            num_samples=4, width=1024, height=1024, num_inference_steps=30, seed=42
        )
        grid = image_grid(images, 2, 2)
        save_path = os.path.join(args.save_dir, 'wo_lora_'+os.path.basename(image_path))
        grid.save(save_path)
        print(f"result has saved in {save_path} ......")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input image dir",
                        default="./data/all_test_data")
    parser.add_argument("--save_dir", type=str, default="./data/all_test_data_xl_faceid_plus_v2")
    parser.add_argument("--faceid_lora_weight", type=float, default=0.8)
    parser.add_argument("--ip_scale", type=float, default=0.8)
    parser.add_argument("--s_scale", type=float, default=1.0)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
