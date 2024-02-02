import torch
from PIL import Image
import cv2
import argparse
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from my_script.util.util import FaceidAcquirer, image_grid
# from my_script.unetfix import CostomUNet2DConditionModel
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
from ip_adapter.utils import register_cross_attention_hook, get_net_attn_map, attnmaps2images


def main(args):
    source_dir = r'/mnt/nfs/file_server/public/mingjiahui/models'
    base_model_path = fr'{source_dir}/Lykon--DreamShaper/'      # "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = fr"{source_dir}/stabilityai--sd-vae-ft-mse"
    ip_ckpt = fr"{source_dir}/h94--IP-Adapter/faceid/ip-adapter-faceid_sd15.bin"
    lora = f"{source_dir}/h94--IP-Adapter/faceid/ip-adapter-faceid_sd15_lora.safetensors"
    device = "cuda"

    app = FaceidAcquirer()

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
    # unet = CostomUNet2DConditionModel.from_pretrained(
    #     base_model_path,
    #     subfolder="unet",
    #     torch_dtype=torch.float16,
    # )
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

    # jiahui'S modify       load lora
    # # ori
    # pipe.load_lora_weights(lora)
    # pipe.load_lora_weights(
    #     f"{source_dir}/h94--IP-Adapter/faceid", 
    #     weight_name="ip-adapter-faceid_sd15_lora.safetensors", 
    #     adapter_name="style"
    #     )
    # pipe.set_adapters(["style"], adapter_weights=[1.0])

    # load ip-adapter
    ip_model = IPAdapterFaceID(pipe, ip_ckpt, device)

    # generate image
    prompt = "closeup photo of a man wearing a white shirt in a garden, high quality, diffuse light, highly detailed, 4k"
    negative_prompt = "blurry, malformed, distorted, naked"

    image_path = args.input
    faceid_embeds, align_face = app.get_face_embeds(cv2.imread(image_path))
 
    images = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=args.batch, width=512, height=512, num_inference_steps=30, seed=42, guidance_scale=6,
    )
    grid = image_grid(images, int(args.batch**0.5), int(args.batch**0.5))
    save_path = os.path.join(args.output, f"{os.path.basename(ip_ckpt).split('.')[0]}_{os.path.basename(args.input)}")
    grid.save(save_path)
    print(f"result has saved in {save_path}")

    if args.visual_atten_map:
        attn_maps = get_net_attn_map((512, 512))
        print(attn_maps.shape)      # {4, 512, 512}
        attn_hot = attnmaps2images(attn_maps)
        import matplotlib.pyplot as plt
        #axes[0].imshow(attn_hot[0], cmap='gray')
        display_images = [cv2.cvtColor(align_face, cv2.COLOR_BGR2RGB)] + attn_hot + [images[0]]
        fig, axes = plt.subplots(1, len(display_images), figsize=(12, 4))
        for axe, image in zip(axes, display_images):
            axe.imshow(image, cmap='gray')
            axe.axis('off')
        # plt.show()
        plt.savefig('./data/other/debug.jpg')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch", type=float, default=1)
    parser.add_argument("--visual_atten_map", action="store_true")
    args = parser.parse_args()
    args.output = os.path.dirname(args.input) + '_output' \
        if args.output is None else args.output
    os.makedirs(args.output, exist_ok=True)
    print(args.output)
    main(args)