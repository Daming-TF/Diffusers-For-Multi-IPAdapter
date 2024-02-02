
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import argparse
from torchvision import transforms

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from my_script.util.util import image_grid
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL, USE_DAFAULT_ATTN
from ip_adapter.attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor
from ip_adapter.utils import is_torch2_available
from ip_adapter.utils import register_cross_attention_hook, get_net_attn_map, attnmaps2images


if is_torch2_available() and (not USE_DAFAULT_ATTN):
    from ip_adapter.attention_processor_faceid import (
        LoRAAttnProcessor2_0 as LoRAAttnProcessor,
    )
    from ip_adapter.attention_processor_faceid import (
        LoRAIPAttnProcessor2_0 as LoRAIPAttnProcessor,
    )
class CostomIPAdapterFaceIDXL(IPAdapterFaceIDXL):
    def set_lora_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, LoRAAttnProcessor) or isinstance(attn_processor, LoRAIPAttnProcessor):
                attn_processor.lora_scale = scale


transform = transforms.Compose([
    transforms.Resize(1024),
    transforms.CenterCrop(1024),
])

def main(args):
    # 1. get face embeds
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    image = cv2.imread(args.input)        # r"./data/all_test_data/wangbaoqiang.jpg"
    faces = app.get(image)

    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

    # 2. load sdxl
    base_model_path = r"/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/"       # "SG161222/RealVisXL_V3.0"
    ip_ckpt = r"/mnt/nfs/file_server/public/mingjiahui/models/h94--IP-Adapter/faceid/ip-adapter-faceid_sdxl.bin"
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
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        add_watermarker=False,
    )
    if args.visual_atten_map:
        pipe.unet = register_cross_attention_hook(pipe.unet)

    # 3. load ip-adapter
    ip_model = CostomIPAdapterFaceIDXL(pipe, ip_ckpt, device)
    ip_model.set_lora_scale(args.faceid_lora_weight)

    # 4. generate image
    prompt = "closeup photo of a man wearing a white shirt in a garden, high quality, diffuse light, highly detailed, 4k"  \
        if args.prompt is None else args.prompt
    negative_prompt = "blurry, malformed, distorted, naked" \
        if args.negative_prompt is None else args.negative_prompt

    images = ip_model.generate(
        prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=args.batch,
        width=1024, height=1024,
        num_inference_steps=30, guidance_scale=7.5, seed=42     # 5.0
    )
    grid = image_grid(images, rows=int(args.batch**0.5), cols=int(args.batch**0.5))
    # image_dir_name = os.path.basename(os.path.dirname(args.input))
    # save_dir = os.path.join(args.output, image_dir_name)
    # os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(args.output, os.path.basename(args.input))
    grid.save(save_path)
    print(f"result has saved in ==> {save_path}")

    if args.visual_atten_map:
        attn_maps = get_net_attn_map((1024, 1024))
        print(attn_maps.shape)
        attn_hot = attnmaps2images(attn_maps)

        import matplotlib.pyplot as plt
        #axes[0].imshow(attn_hot[0], cmap='gray')
        display_images = [transform(Image.open(args.input))] + attn_hot + [images[0]]
        fig, axes = plt.subplots(1, len(display_images), figsize=(12, 4))
        for axe, image in zip(axes, display_images):
            axe.imshow(image, cmap='gray')
            axe.axis('off')
        # plt.show()
        plt.savefig('./data/other/debug.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="input image path")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--ip_scale", type=float, default=0.7)
    parser.add_argument("--faceid_lora_weight", type=float, default=0.7)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--visual_atten_map", action="store_true")
    args = parser.parse_args()
    args.output = os.path.dirname(args.input) + '_output' \
        if args.output is None else args.output
    os.makedirs(args.output, exist_ok=True)
    main(args)
