
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
import cv2
from insightface.app import FaceAnalysis
import numpy as np
import argparse
import json
from tqdm import tqdm
import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID, IPAdapterFaceIDXL
from my_script.util.transfer_ckpt import transfer_ckpt


class FaceAcquirer:
    def __init__(self, ):
        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))      # det_size=(640, 640)
    
    def get_face_embeds(self, image: np.ndarray):
        faces = self.app.get(image)
        return torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=r"/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data")
    # parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--num_tokens", type=int, default=4)
    args = parser.parse_args()
    # args.output = os.path.join(r'./output/result', os.path.basename(args.input_dir)) if args.output is None else args.output
    transfer_ckpt(args.ckpt_dir) if not os.path.exists(os.path.join(args.ckpt_dir, "sdxl_faceid.bin")) else None
    base_model_path = "/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/"
    device = "cuda"

    save_dir = os.path.join(args.ckpt_dir, 'test_sampling')
    if os.path.exists(save_dir):
        print("test_sampling dir is exists")
        exit(0)
    os.makedirs(save_dir, exist_ok=True)

    # get image path
    image_paths = []
    # if args.input_dir is None:
    #     json_path = r'./data/train_data/1face_debug.json'
    #     with open(json_path, 'r')as f:
    #         data = json.load(f)
    #     for metadata in data:
    #         image_paths.append(metadata['image_file'])
    #         print(metadata['image_file'])
    # else:
    image_paths = [os.path.join(args.input_dir, name) for name in os.listdir(args.input_dir) if name.endswith('.jpg')]

    # load sdXL
    print("loading model......")
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
        scheduler=noise_scheduler,
        torch_dtype=torch.float16,
        add_watermarker=False,
    )

    # load ip-adapter
    ip_ckpt = os.path.join(args.ckpt_dir, "sdxl_faceid.bin")
    ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device, num_tokens=args.num_tokens)
    
    # initial face 
    face_acquirer = FaceAcquirer()

    # generate image
    for image_path in tqdm(image_paths):
        # faceid image
        image = cv2.imread(image_path)
        faceid_embeds = face_acquirer.get_face_embeds(image)
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
            num_samples=1,  # 4
            width=1024,
            height=1024,     # 768
            num_inference_steps=30, 
            seed=2023
        )[0]

        # save
        save_name = os.path.basename(image_path)
        save_path = os.path.join(save_dir, save_name)
        image.save(save_path)
        print(f"result has saved in {save_path}")


if __name__ == '__main__':
    main()
