import argparse
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
import subprocess
import torch
from torchvision import transforms
import logging
import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))


test0_data_dir = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data/"
test0_data_paths = [os.path.join(test0_data_dir, name)for name in os.listdir(test0_data_dir)\
                    if not name.endswith('.txt')]
test1_data_dir = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/test_data_V2/"
test1_data_dirs_ = [os.path.join(test1_data_dir, dir_name) for dir_name in os.listdir(test1_data_dir)]
test1_data_paths = []
for test1_data_dir_ in test1_data_dirs_:
    test1_data_paths += [os.path.join(test1_data_dir_, name)for name in os.listdir(test1_data_dir_)\
                        if not name.endswith('.txt')]
test_data_paths = test0_data_paths + test1_data_paths
# test_data_paths = test_data_paths[:5]
transform = transforms.Resize(1024)


def crop_face_image(image: Image.Image, bbox, factor=2):
    w, h = image.size
    min_size = min(w, h)
    # resize_ratio = min_size /1024
    image = transform(image)
    w, h = image.size
    l_top_x, l_top_y, r_bottom_x, r_bottom_y = bbox
    l_top_x = max(0, l_top_x)
    l_top_y = max(0, l_top_y)
    r_bottom_x = min(w, r_bottom_x)
    r_bottom_y = min(h, r_bottom_y)

    b_w, b_h = (r_bottom_x - l_top_x), (r_bottom_y - l_top_y)
    center_x = l_top_x + b_w//2
    center_y = l_top_y + b_h//2

    # new_size = min(min(b_w, b_h) * factor, min(w, h))
    new_size = max(b_w, b_h) * factor
    x_start = max(0, center_x-new_size//2)
    y_start = max(0, center_y-new_size//2)
    x_end = min(w, center_x+new_size//2)
    y_end = min(h, center_y+new_size//2)
    if x_start==0 or y_start==0:
        x_end = min(w, x_start+new_size)
        y_end = min(h, y_start+new_size)
    elif x_end==w or y_end==h:
        x_start = max(0, x_end-new_size)
        y_start = max(0, y_end-new_size)
    crop_image = Image.fromarray(np.array(image)[int(y_start):int(y_end), int(x_start):int(x_end)])
    return crop_image


def inference(checkpoint_dirs, ckpt_name):
    if not isinstance(checkpoint_dirs, list):
        checkpoint_dirs = [checkpoint_dirs]
    from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID
    from my_script.util.transfer_ckpt import transfer_ckpt
    from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
    from my_script.util.util import FaceidAcquirer, image_grid
    app = FaceidAcquirer()
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
        safety_checker=None,
    )

    # 4. process
    for checkpoint_dir in tqdm(checkpoint_dirs):
        print(f"test0 num:{len(test0_data_paths)}\ttest1 num:{len(test1_data_paths)}\ttotal num:{len(test_data_paths)}")
        # 4.2 transfer ckpt file
        if not os.path.exists(os.path.join(checkpoint_dir, ckpt_name)):
            transfer_ckpt(checkpoint_dir, output_name=ckpt_name) 
        output_dir = os.path.join(checkpoint_dir, 'test_sampling')
        os.makedirs(output_dir, exist_ok=True)

        # 4.4 load ip-adapter
        ip_ckpt = os.path.join(checkpoint_dir, ckpt_name)
        ip_model = IPAdapterFaceID(pipe, ip_ckpt, device, num_tokens=16, n_cond=1)

        # 4.5 generate image
        for image_path in tqdm(test_data_paths):
            save_name = os.path.basename(image_path)
            save_path = os.path.join(output_dir, save_name)
            if os.path.exists(save_path):
                continue
            faceid_embeds = app.get_multi_embeds(image_path)
            # prompt
            suffix = os.path.basename(image_path).split('.')[-1]
            txt_path = image_path.replace(suffix, 'txt')
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
            image.save(save_path)
            print(f"image result has saved in {save_path}")


def inference_ti_token(checkpoint_dirs, ckpt_name):
    if not isinstance(checkpoint_dirs, list):
        checkpoint_dirs = [checkpoint_dirs]
    # from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID
    from my_script.models.IPAdapter_face import IPAdapterFaceID_TiToken
    from my_script.util.transfer_ckpt import transfer_ckpt
    from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
    from insightface.app import FaceAnalysis
    # from my_script.util.util import FaceidAcquirer, image_grid
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    print("loading model......")
    source_dir = '/mnt/nfs/file_server/public/mingjiahui/models'
    base_model_path = f"{source_dir}/SG161222--Realistic_Vision_V4.0_noVAE/"
    vae_model_path = fr"{source_dir}/stabilityai--sd-vae-ft-mse"
    image_encoder_path = f"{source_dir}/h94--IP-Adapter/h94--IP-Adapter/models/image_encoder"
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
        safety_checker=None,
    )

    # 4. process
    for checkpoint_dir in tqdm(checkpoint_dirs):
        print(f"test0 num:{len(test0_data_paths)}\ttest1 num:{len(test1_data_paths)}\ttotal num:{len(test_data_paths)}")
        # 4.2 transfer ckpt file
        if not os.path.exists(os.path.join(checkpoint_dir, ckpt_name)):
            transfer_ckpt(checkpoint_dir, output_name=ckpt_name) 
        output_dir = os.path.join(checkpoint_dir, 'test_sampling')
        os.makedirs(output_dir, exist_ok=True)

        # 4.4 load ip-adapter
        ip_ckpt = os.path.join(checkpoint_dir, ckpt_name)
        ip_model = IPAdapterFaceID_TiToken(
            pipe, device, 
            ip_ckpt, image_encoder_path,
            num_tokens=16, ti_num_tokens=4,
            n_cond=1)

        # 4.5 generate image
        for image_path in tqdm(test_data_paths):
            save_name = os.path.basename(image_path)
            save_path = os.path.join(output_dir, save_name)
            if os.path.exists(save_path):
                continue
            # faceid_embeds = app.get_multi_embeds(image_path)
            image = transform(Image.open(image_path).convert("RGB"))
            faces = app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            assert len(faces) != 0, ValueError("detect no face")
            face = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]
            faceid_embeds = torch.from_numpy(face.normed_embedding).unsqueeze(0)
            face_image = crop_face_image(image, face.bbox)
            # print(f"faceid_embeds:{type(faceid_embeds)}")
            # print(f"faceid_embeds:{faceid_embeds.shape}")
            # print(f"face_image:{type(face_image)}")
        
            # prompt
            suffix = os.path.basename(image_path).split('.')[-1]
            txt_path = image_path.replace(suffix, 'txt')
            with open(txt_path, 'r')as f:
                lines = f.readlines()
            assert len(lines) == 1
            prompt = lines[0]
            # processing
            image = ip_model.generate(
                prompt=prompt,
                crop_images=face_image,
                faceid_embeds=faceid_embeds, 
                num_samples=1, 
                width=512, height=512, 
                num_inference_steps=30, 
                seed=42, 
                guidance_scale=6,
            )[0]

            # save
            image.save(save_path)
            print(f"image result has saved in {save_path}")


def inference_instantid(checkpoint_dir, ckpt_name, controlnet=None):
    from insightface.app import FaceAnalysis
    from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel
    from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID
    from InstantID.pipeline_stable_diffusion_xl_instantid import draw_kps
    from InstantID.infer import resize_img
    from my_script.util.transfer_ckpt import transfer_ckpt
    # from my_script.util.util import FaceidAcquirer, image_grid
    from my_script.models.InstandID import StableDiffusionControlNetPipelineCostomInstantID, InstantIDFaceID
    app = FaceAnalysis(name='/home/mingjiahui/.insightface/models/antelopev2/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

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
    controlnet = ControlNetModel.from_pretrained(checkpoint_dir, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipelineCostomInstantID.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        # unet=unet,
        vae=vae,
        controlnet=controlnet,
        feature_extractor=None,
        safety_checker=None,
    )


    print(f"test0 num:{len(test0_data_paths)}\ttest1 num:{len(test1_data_paths)}\ttotal num:{len(test_data_paths)}")
    # 4.2 transfer ckpt file
    if not os.path.exists(os.path.join(checkpoint_dir, ckpt_name)):
        transfer_ckpt(checkpoint_dir, output_name=ckpt_name) 
    output_dir = os.path.join(checkpoint_dir, 'test_sampling')
    os.makedirs(output_dir, exist_ok=True)

    # 4.4 load ip-adapter
    ip_ckpt = os.path.join(checkpoint_dir, ckpt_name)
    ip_model = InstantIDFaceID(pipe, ip_ckpt, device, num_tokens=16, n_cond=1)

    # 4.5 generate image
    for image_path in tqdm(test_data_paths):
        save_name = os.path.basename(image_path)
        save_path = os.path.join(output_dir, save_name)
        # if os.path.exists(save_path):
        #     continue
        # face info
        face_image = Image.open(image_path).convert("RGB")
        face_image = resize_img(face_image)
        face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]   # only use the maximum face
        face_emb = torch.from_numpy(face_info.normed_embedding).unsqueeze(0).unsqueeze(0)
        face_kps = draw_kps(face_image, face_info['kps'])
        # prompt
        suffix = os.path.basename(image_path).split('.')[-1]
        txt_path = image_path.replace(suffix, 'txt')
        with open(txt_path, 'r')as f:
            lines = f.readlines()
        assert len(lines) == 1
        prompt = lines[0]
        # processing
        image = ip_model.generate(
            prompt=prompt,
            num_samples=1, 
            width=512, height=512, 
            num_inference_steps=30, 
            seed=42, 
            guidance_scale=6,
            faceid_embeds=face_emb, image=face_kps,
        )[0]

        # save
        image.save(save_path)
        print(f"image result has saved in {save_path}")


def distance(checkpoint_dirs):
    logging.basicConfig(level=logging.ERROR)
    if not isinstance(checkpoint_dirs, list):
        checkpoint_dirs = [checkpoint_dirs]
    print(f"test0 num:{len(test0_data_paths)}\ttest1 num:{len(test1_data_paths)}\ttotal num:{len(test_data_paths)}")
    from deepface import DeepFace
    from data.xlsx_writer import WriteExcel
    cuda_devices=os.environ.get('CUDA_VISIBLE_DEVICES', '-1')
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    test_models = ["Facenet512", "SFace", "ArcFace", "VGG-Face"]
    test_data_ids = [os.path.basename(test_data_path).split('.')[0] for test_data_path in test_data_paths]
    for checkpoint_dir in tqdm(checkpoint_dirs):
        save_path = os.path.join(checkpoint_dir, 'result.xlsx')
        xlsx_writer = WriteExcel(save_path, test_data_ids, test_models)
        output_dir = os.path.join(checkpoint_dir, 'test_sampling')
        if not os.path.exists(output_dir) or os.path.exists(save_path):
            continue
        output_names = [name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name))]
        # assert len(test_data_paths)==len(output_names), \
        #     ValueError(f"test_data num:{len(test_data_paths)}\tgen image num:{len(output_names)}")
        for i, model in enumerate(test_models):
            col_result = []
            for test_data_path in tqdm(test_data_paths):
                image_name = os.path.basename(test_data_path)
                output_data_path = os.path.join(output_dir, image_name)
                if not os.path.exists(output_data_path):
                    continue
                # assert os.path.exists(output_data_path), ValueError(f"{output_data_path} is not exists")
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
                result_ = DeepFace.verify(img1_path = output_data_path, 
                    img2_path = test_data_path, 
                    model_name=model,
                    detector_backend="mtcnn"
                )['distance']
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__
                print(f"{os.path.basename(checkpoint_dir)}\t{image_name}\t{result_}")
                col_result.append(result_)
            xlsx_writer.write(col_result, i)
        xlsx_writer.close()
        print(f"xlsx result has saved in {save_path}")
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", type=str, nargs='+',
        default=None)
    parser.add_argument("--num_tokens", type=int, default=16)
    parser.add_argument("--ckpt_name", type=str, default='sd15_faceid_portrait.bin')
    parser.add_argument("--save_name", type=str, default='test_sampling')
    parser.add_argument("--mode", type=str, default='distance',help="Union['inference', 'distance']")
    args = parser.parse_args()
    # 1.init
    source_dir = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/finetune/base-portrait"
    reset_ckpt_input = [
        f"{source_dir}/20140130-sd15-crop--V1-wo_xformer-scratch",
        f"{source_dir}/20140130-sd15-crop--V1-wo_xformer-scratch_from_step6000",
        f"{source_dir}/20140205-sd15-crop--V1-wo_xformer-scratch_from_step160000",
        f"{source_dir}/20140131-sd15-crop--V1-wo_xformer-scratch_from_step26000/",
        f"{source_dir}/20140205-sd15-crop--V1-wo_xformer-scratch_from_step190000",
    ]
    args.input_dirs = reset_ckpt_input if args.input_dirs is None else args.input_dirs
    print(f"input_dirs:{args.input_dirs}")
    # 3. get ckpt paths
    checkpoint_dirs = []
    if isinstance(args.input_dirs, list):
        for input_dir in args.input_dirs:
            if 'checkpoint' not in os.path.basename(input_dir):
                checkpoint_dirs += [os.path.join(input_dir, name) for name in os.listdir(input_dir)]
            else:
                checkpoint_dirs += [input_dir]
    else:
        if 'checkpoint' not in os.path.basename(args.input_dir):
            checkpoint_dirs += [os.path.join(input_dir, name) for name in os.listdir(input_dir)]
        else:
            checkpoint_dirs = [args.input_dirs]
    print(f"**check:{checkpoint_dirs[:5]}")
    def extract_number(input):
        dir_name = os.path.basename(os.path.dirname(input))
        pretrain_step = int(dir_name.split('step')[-1]) if 'step' in dir_name else 0
        fineturn_step = int(os.path.basename(input).split('-')[-1])
        return pretrain_step+fineturn_step
    checkpoint_dirs = sorted(checkpoint_dirs, key=extract_number)
    for checkpoint_dir in checkpoint_dirs:
        print(checkpoint_dir)

    if args.mode == 'inference':
        inference(checkpoint_dirs, args.ckpt_name)
    elif args.mode == 'distance':
        distance(checkpoint_dirs)
    elif args.mode == 'inference_ti':
        inference_ti_token(checkpoint_dirs, args.ckpt_name)
    else:
        ValueError("The mode param must be selected between inference and distance")
