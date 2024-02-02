
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
import cv2
# from insightface.app import FaceAnalysis
import numpy as np
import argparse
import json
from tqdm import tqdm
import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID
from my_script.util.transfer_ckpt import transfer_ckpt
from my_script.util.util import FaceidAcquirer, image_grid
from ip_adapter.utils import register_cross_attention_hook, get_net_attn_map, attnmaps2images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=r"/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/all_test_data/")
    # parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--num_tokens", type=int, default=16)
    parser.add_argument("--ckpt_name", type=str, default='sd15_faceid_portrait.bin')
    parser.add_argument("--save_name", type=str, default='test_sampling')
    parser.add_argument("--visual_atten_map", action="store_true")
    args = parser.parse_args()
    # args.output = os.path.join(r'./output/result', os.path.basename(args.input_dir)) if args.output is None else args.output
    if not os.path.exists(os.path.join(args.ckpt_dir, args.ckpt_name)):
        transfer_ckpt(args.ckpt_dir, output_name=args.ckpt_name) 
    device = "cuda"

    save_dir = os.path.join(args.ckpt_dir, args.save_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"dir1 num:{len(os.listdir(save_dir))}\t dir2 num:{len([name for name in os.listdir(args.input_dir)if name.endswith('.jpg')])}")
    # if os.path.exists(save_dir) and len(os.listdir(save_dir))==len(os.listdir(args.input_dir)):
    #     print("test_sampling dir is exists")
    #     exit(0)
    os.makedirs(save_dir, exist_ok=True)

    # get image path
    image_paths = []
    image_paths = [os.path.join(args.input_dir, name) for name in os.listdir(args.input_dir) if name.endswith('.jpg')]

    # load sd15
    print("loading model......")
    source_dir = '/mnt/nfs/file_server/public/mingjiahui/models'
    base_model_path = f"{source_dir}/SG161222--Realistic_Vision_V4.0_noVAE/"
    vae_model_path = fr"{source_dir}/stabilityai--sd-vae-ft-mse"
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
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        # unet=unet,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )
    if args.visual_atten_map:
        print("register hook......")
        pipe.unet = register_cross_attention_hook(pipe.unet)
        print('finish!')

    # load ip-adapter
    ip_ckpt = os.path.join(args.ckpt_dir, "sd15_faceid_portrait.bin")
    ip_model = IPAdapterFaceID(pipe, ip_ckpt, device, num_tokens=16, n_cond=1)

    # initial face 
    app = FaceidAcquirer()

    # generate image
    for image_path in tqdm(image_paths):
        faceid_embeds = app.get_multi_embeds(image_path)
        # prompt
        txt_path = image_path.replace('.jpg', '.txt')
        with open(txt_path, 'r')as f:
            lines = f.readlines()
        assert len(lines) == 1
        prompt = lines[0]
        # processing
        image = ip_model.generate(
            prompt=prompt,
            faceid_embeds=faceid_embeds, 
            num_samples=1, 
            width=512, height=512, 
            num_inference_steps=30, 
            seed=42, 
            guidance_scale=6,
        )[0]

        # save
        save_name = os.path.basename(image_path)
        save_path = os.path.join(save_dir, save_name)
        image.save(save_path)
        print(f"result has saved in {save_path}")

        if args.visual_atten_map:
            save_dir_ = os.path.join(save_dir, 'attn_map')
            image_id = save_name.split('.')[0]
            os.makedirs(save_dir_, exist_ok=True)
            # per layer per head hot map
            attn_maps, attn_maps_mean = get_net_attn_map((512, 512))        # {16,8,512,512} # {16,512,512} 
            # print(attn_maps.shape)      # {4, 512, 512}  after modify {16,8,512,512}
            # attn_hot = attnmaps2images(attn_maps)
            result = None
            for layer_attn_maps in attn_maps:      # {8,512,512}
                attn_hot = attnmaps2images(layer_attn_maps)
                layer_attn_output = None
                for attn_hot_ in attn_hot:
                    layer_attn_output = cv2.hconcat([layer_attn_output, np.array(attn_hot_)]) if layer_attn_output is not None else np.array(attn_hot_)
                result = cv2.vconcat([result, layer_attn_output]) if result is not None else layer_attn_output
            
            if len(result.shape) == 2:
                result = np.expand_dims(result, axis=2)
                result = np.repeat(result, 3, axis=2)      
            save_path = os.path.join(save_dir_, f"{image_id}_8head_16layers.jpg") 
            Image.fromarray(result).save(save_path)

            attn_hot = attnmaps2images(attn_maps_mean)
            _, align_face = app.get_face_embeds(Image.open(image_path))
            result = cv2.resize(cv2.cvtColor(align_face, cv2.COLOR_BGR2RGB), (512,512))
            for attn_hot_ in attn_hot:
                attn_hot_ = cv2.cvtColor(np.array(attn_hot_), cv2.COLOR_GRAY2RGB)
                result = cv2.hconcat([result, attn_hot_])
            result = cv2.hconcat([result, np.array(image)])
            save_path = os.path.join(save_dir_, f"{image_id}.jpg") 
            Image.fromarray(result).save(save_path)


if __name__ == '__main__':
    main()
