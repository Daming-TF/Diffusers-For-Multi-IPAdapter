import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
import numpy as np
import cv2
import os
import sys
import argparse
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from my_script.util.util import FaceidAcquirer, image_grid
from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID
from ip_adapter.utils import register_cross_attention_hook, get_net_attn_map, attnmaps2images


def main(args):
    source_dir = r'/mnt/nfs/file_server/public/mingjiahui/models'
    base_model_path =f"{source_dir}/SG161222--Realistic_Vision_V4.0_noVAE/"   # "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = f"{source_dir}/stabilityai--sd-vae-ft-mse/"   # "stabilityai/sd-vae-ft-mse"
    ip_ckpt = f"{source_dir}/h94--IP-Adapter/h94--IP-Adapter/models/ip-adapter-faceid-portrait_sd15.bin"
    device = "cuda"

    if args.embeds_path is None:
        assert args.input is not None
        image_paths = [os.path.join(args.input, name)for name in os.listdir(args.input)][:5] if os.path.isdir(args.input) \
        else [args.input]
        app = FaceidAcquirer()
        faceid_embeds = app.get_multi_embeds(image_paths)
        n_cond = faceid_embeds.shape[1]
    else:
        print(f"loading embeds path ==>{args.embeds_path}")
        faceid_embeds = np.load(args.embeds_path)
        if len(faceid_embeds.shape) == 2 and len(faceid_embeds.shape) != 3:
            faceid_embeds = torch.from_numpy(faceid_embeds).unsqueeze(0)
        else:
            ValueError(f"faceid_embeds shape is error ==> {faceid_embeds.shape}")
        n_cond = faceid_embeds.shape[1]
    print(faceid_embeds.shape)

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
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )
    if args.visual_atten_map:
        print("register hook......")
        pipe.unet = register_cross_attention_hook(pipe.unet)
        print('finish!')

    # load ip-adapter
    ip_model = IPAdapterFaceID(pipe, ip_ckpt, device, num_tokens=16, n_cond=n_cond)

    if args.embeds_path is None:
        assert len(image_paths)==1
        suffix = os.path.basename(image_paths[0]).split('.')[1]
        txt_path = image_paths[0].replace(suffix, 'txt')
    else:
        txt_path = args.embeds_path.replace('npy', 'txt')
    
    if os.path.exists(txt_path):
        with open(txt_path, 'r')as f:
            prompt = f.readlines()[0]
    prompt = prompt if args.prompt is None else args.prompt
    # negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

    print(f"prompt:{prompt}\n negative_prompt:{args.negative_prompt}")
    images = ip_model.generate(
        prompt=prompt, 
        # negative_prompt=negative_prompt, 
        faceid_embeds=faceid_embeds, 
        num_samples=args.batch, 
        width=512, height=512, 
        num_inference_steps=30, 
        seed=2023
    )
    grid = image_grid(images, int(args.batch**0.5), int(args.batch**0.5))
    grid.save(args.save_path)
    print(f"result has saved in {args.save_path}")

    if args.visual_atten_map:
        # per layer per head hot map
        attn_maps, attn_maps_mean = get_net_attn_map((512, 512))        # {16,8,512,512} # {16,512,512} 
        print(attn_maps.shape)      # {4, 512, 512}  after modify {16,8,512,512}
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
        Image.fromarray(result).save('/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/other/debug_visual_attn_map_group.jpg')

        attn_hot = attnmaps2images(attn_maps_mean)
        _, align_face = app.get_face_embeds(Image.open(image_paths[0]))
        result = cv2.resize(cv2.cvtColor(align_face, cv2.COLOR_BGR2RGB), (512,512))
        for attn_hot_ in attn_hot:
            attn_hot_ = cv2.cvtColor(np.array(attn_hot_), cv2.COLOR_GRAY2RGB)
            result = cv2.hconcat([result, attn_hot_])
        result = cv2.hconcat([result, np.array(images[0])])
        Image.fromarray(result).save('/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/other/debug_visual_attn_map.jpg')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="image_path or image_dir for 1 portrait")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--embeds_path", type=str, default=None)
    parser.add_argument("--visual_atten_map", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    if args.input is not None:
        if os.path.isdir(args.input):
            args.save_path = os.path.join(os.path.dirname(os.path.dirname(args.input))+'_output', 'faceid_portrait.jpg') if args.save_path is None else args.save_path
        else:
            args.save_path = os.path.join(os.path.dirname(args.input)+'_output', os.path.basename(args.input)) if args.save_path is None else args.save_path
    elif args.embeds_path is not None:
        args.save_path = os.path.join(os.path.dirname(args.embeds_path)+'_output', os.path.basename(args.embeds_path).split('.')[0]+'.jpg') if args.save_path is None else args.save_path
    else:
        ValueError("'--embeds_path' and '--input' cannot both be N0one")
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    print(f"result will save in ==> {args.save_path}")
    main(args)
