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
                    if not name.endswith('.txt') and 'temp' not in name]
test1_data_dir = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/test_data_V2/"
test1_data_dirs_ = [os.path.join(test1_data_dir, dir_name) for dir_name in os.listdir(test1_data_dir)]
test1_data_paths = []
for test1_data_dir_ in test1_data_dirs_:
    test1_data_paths += [os.path.join(test1_data_dir_, name)for name in os.listdir(test1_data_dir_)\
                        if not name.endswith('.txt') and 'temp' not in name]
test_data_paths = test0_data_paths + test1_data_paths
# test_data_paths = test_data_paths[:5]
# print(test_data_paths)
# exit(0)
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


def inference(checkpoint_dirs, ckpt_name, image_encoder='buffalo_l', output_dir=None):
    if not isinstance(checkpoint_dirs, list):
        checkpoint_dirs = [checkpoint_dirs]
    from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID
    from my_script.util.transfer_ckpt import transfer_ckpt
    from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
    from my_script.util.util import FaceidAcquirer, image_grid
    app = FaceidAcquirer(name=image_encoder)
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
        output_dir = os.path.join(checkpoint_dir, 'test_sampling') if output_dir is None else output_dir
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


def inference_ti_token(checkpoint_dirs, ckpt_name, output_dir=None):
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
        output_dir = os.path.join(checkpoint_dir, 'test_sampling') if output_dir is None else output_dir
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
            image = Image.open(image_path).convert("RGB")
            image = transform(image)
            faces = app.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
            assert len(faces) != 0, ValueError(f"detect no face ==> {image_path}")
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


def inference_instantid(checkpoint_dir, ckpt_name, resampler=True, num_tokens=16, output_dir=None):
    from insightface.app import FaceAnalysis
    from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel
    from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID
    from InstantID.pipeline_stable_diffusion_xl_instantid import draw_kps
    from InstantID.infer import resize_img
    from my_script.util.transfer_ckpt import transfer_ckpt
    # from my_script.util.util import FaceidAcquirer, image_grid
    from my_script.models.InstandID import StableDiffusionControlNetPipelineCostomInstantID, InstantIDFaceID
    app = FaceAnalysis(name='/home/mingjiahui/.insightface/models/buffalo_l/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
    ])

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
    output_dir = os.path.join(checkpoint_dir, 'test_sampling') if output_dir is None else output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 4.4 load ip-adapter
    ip_ckpt = os.path.join(checkpoint_dir, ckpt_name)
    ip_model = InstantIDFaceID(pipe, ip_ckpt, device, num_tokens=num_tokens, n_cond=1, resampler=resampler)

    # 4.5 generate image
    for image_path in tqdm(test_data_paths):
        # if os.path.exists(save_path):
        #     continue
        # face info
        face_image = Image.open(image_path).convert("RGB")
        # face_image = resize_img(face_image)
        face_image = transform(face_image)
        face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        if len(face_info) == 0:
            print(f"no face find ==> {image_path}")
            continue
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]   # only use the maximum face
        face_emb = torch.from_numpy(face_info.normed_embedding).unsqueeze(0).unsqueeze(0)

        cropped_img, kps = resize_and_crop(face_image, face_info['kps'], face_info['bbox'].tolist(), factor=2)
        cropped_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        # face_kps = draw_kps(face_image, face_info['kps'])
        face_kps = draw_kps(cropped_img, kps)

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
        save_name = os.path.basename(image_path)
        prefix, suffix = save_name.split('.')
        save_path_0 = os.path.join(output_dir, save_name)
        save_path_1 = os.path.join(output_dir, prefix+'-info.'+suffix)
        info = Image.fromarray(cv2.hconcat([np.array(image), np.array(cropped_img), np.array(face_kps)]))
        image.save(save_path_0)
        info.save(save_path_1)
        print(f"image result has saved in {save_path_0}")


def inference_styleGAN(checkpoint_dirs, ckpt_name, image_encoder='buffalo_l', sr=False):
    if not isinstance(checkpoint_dirs, list):
        checkpoint_dirs = [checkpoint_dirs]
    from insightface.app import FaceAnalysis
    from my_script.util.transfer_ckpt import transfer_ckpt
    from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
    from my_script.wplus_adapter.models import WPlusAdapter

    # 0. init
    wplus_source_dir = "/home/mingjiahui/projects/w-plus-adapter"
    device = "cuda"
    sys.path.append(wplus_source_dir)
    sys.path.append(os.path.join(wplus_source_dir, 'my_script'))
    sys.path.append(os.path.join(wplus_source_dir, 'script'))
    from align_face import norm_crop
    from utils import tensor2pil, pil2tensor, color_parse_map
    
    # 1.1 init facd det model
    app = FaceAnalysis(name='/home/mingjiahui/.insightface/models/buffalo_l/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    app_trans = transforms.Resize(512)
    # 1.2 init rm bg model
    from script.models.parsenet import ParseNet
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    parse_net = ParseNet(512, 512, 32, 64, 19, norm_type='bn', relu_type='LeakyReLU', ch_range=[32, 256])
    parse_net.eval()
    parse_net.load_state_dict(torch.load(os.path.join(wplus_source_dir, \
        f'./script/weights/parse_multi_iter_90000.pth')))
    # 1.3 init sr model
    from script.models.bfrnet import PSFRGenerator
    bfr_net = PSFRGenerator(3, 3, in_size=512, out_size=512, relu_type='LeakyReLU', parse_ch=19, norm_type='spade')
    for m in bfr_net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.utils.spectral_norm(m)
    bfr_net.eval()
    bfr_net.load_state_dict(torch.load(os.path.join(wplus_source_dir, \
        './script/weights/psfrgan_epoch15_net_G.pth')))
    
    # 1.4 init e4e model
    from script.models.psp import pSp
    e4e_path = os.path.join(wplus_source_dir, './script/weights/e4e_ffhq_encode.pt')
    e4e_ckpt = torch.load(e4e_path, map_location='cpu')
    latent_avg = e4e_ckpt['latent_avg'].to(device)
    e4e_opts = e4e_ckpt['opts']
    e4e_opts['checkpoint_path'] = e4e_path
    e4e_opts['device'] = device
    opts = argparse.Namespace(**e4e_opts)
    e4e = pSp(opts).to(device)
    e4e.eval()

    # 1.5 init sd15 
    print("loading model......")
    source_dir = '/mnt/nfs/file_server/public/mingjiahui/models'
    base_model_path = f"{source_dir}/SG161222--Realistic_Vision_V4.0_noVAE/"
    vae_model_path = fr"{source_dir}/stabilityai--sd-vae-ft-mse"
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

    # 2. process
    for checkpoint_dir in tqdm(checkpoint_dirs):
        print(f"test0 num:{len(test0_data_paths)}\ttest1 num:{len(test1_data_paths)}\ttotal num:{len(test_data_paths)}")
        # 2.1 transfer ckpt file
        if not os.path.exists(os.path.join(checkpoint_dir, ckpt_name)):
            transfer_ckpt(checkpoint_dir, output_name=ckpt_name) 
        output_dir = os.path.join(checkpoint_dir, 'test_sampling')
        os.makedirs(output_dir, exist_ok=True)

        # 2.2 load wplus-adapter
        wp_ckpt = os.path.join(checkpoint_dir, ckpt_name)
        wp_model = WPlusAdapter(pipe, wp_ckpt, device)

        # 2.3 generate image
        for image_path in tqdm(test_data_paths):
            image_name = os.path.basename(image_path)
            image_id, suffix = image_name.split('.')

            # Step1 face crop and align
            save_path_step1 = os.path.join(output_dir, image_id+'--align.jpg')
            original_image = Image.open(image_path)
            original_image = original_image.convert("RGB")
            # my align method
            np_original_image = cv2.cvtColor(np.array(transform(original_image)), cv2.COLOR_RGB2BGR)
            faces = app.get(np_original_image)
            assert len(faces) != 0, ValueError("detect no face")
            face = sorted(faces, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
            input_image = norm_crop(np_original_image, landmark=face.kps, image_size=224, factor=1.25)
            input_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
            input_image.save(save_path_step1)

            # Step 2 remove background
            save_path_step2 = os.path.join(output_dir, image_id+'--mask.png')
            input_image = input_image.resize((512, 512), Image.BILINEAR)
            img_tensor = trans(input_image).unsqueeze(0)
            with torch.no_grad():
                parse_map, _ = parse_net(img_tensor)
                if sr:
                    print("using super reso......")
                    parse_map_sm = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
                    output_SR = bfr_net(img_tensor, parse_map_sm)
                    save_img_sr = tensor2pil(output_SR)
                    # result = cv2.hconcat([np.array(input_image.resize((512, 512))), \
                    #             np.array(save_img_sr.resize((512, 512)))])
                    # Image.fromarray(result).save("/home/mingjiahui/projects/w-plus-adapter/output/SR.jpg")
                    input_image = save_img_sr
            
            parse_map_sm = (parse_map == parse_map.max(dim=1, keepdim=True)[0]).float()
            parse_img = color_parse_map(parse_map_sm)

            img_np = parse_img[0]
            img_np = np.mean(img_np, axis=2)
            img_np[img_np>0] = 255
            img_np = 255 - img_np
            input_mask = Image.fromarray(img_np.astype(np.uint8))
            # input_mask.save(save_path_step2)

            # Step3 get w-plus embeds 
            save_path_step3 = os.path.join(output_dir, image_id+'.pt')
            image = pil2tensor(input_image)
            image = (image - 127.5) / 127.5     # Normalize
            kernel_size = 5
            mask_image = cv2.GaussianBlur(np.array(input_mask), (kernel_size, kernel_size), 0)
            mask_image = Image.fromarray(mask_image.astype(np.uint8)) 

            mask_image = mask_image.resize((256, 256))
            mask_image = np.asarray(mask_image).astype(np.float32) # C,H,W -> H,W,C
            mask_image = torch.FloatTensor(mask_image.copy())
            input_mask = mask_image / 255.0
            image = image * (1 - input_mask) + input_mask
            image = image.unsqueeze(0).to(device)

            with torch.no_grad():
                latents_psp = e4e.encoder(image)
            if latents_psp.ndim == 2:
                latents_psp = latents_psp + latent_avg.repeat(latents_psp.shape[0], 1, 1)[:, 0, :]
            else:
                latents_psp = latents_psp + latent_avg.repeat(latents_psp.shape[0], 1, 1)
            wplus_embeds = latents_psp

            # step4: get prompt
            txt_path = image_path.replace(suffix, 'txt')
            with open(txt_path, 'r')as f:
                lines = f.readlines()
            assert len(lines) == 1
            prompt = lines[0]
            # step5: inference
            image = wp_model.generate_idnoise(
                prompt=prompt, 
                w=wplus_embeds.repeat(1, 1, 1).to(device, torch.float16), 
                scale=1.0, 
                num_samples=1, 
                num_inference_steps=30, 
                seed=42, 
                negative_prompt=None
                )[0]

            # save
            save_path = os.path.join(output_dir, image_name)
            image.save(save_path)
            print(f"image result has saved in {save_path}")


def distance(checkpoint_dirs):
    logging.basicConfig(level=logging.ERROR)
    if not isinstance(checkpoint_dirs, list):
        checkpoint_dirs = [checkpoint_dirs]
    print(f"test0 num:{len(test0_data_paths)}\ttest1 num:{len(test1_data_paths)}\ttotal num:{len(test_data_paths)}")
    from deepface import DeepFace
    from data.xlsx_writer import WriteExcel
    # cuda_devices=os.environ.get('CUDA_VISIBLE_DEVICES', '-1')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices


def save_sample_pic(save_path, batch):
    import torchvision.transforms.functional as TF
    images = batch["images"]
    condition_images = batch["condition_images"]
    image_paths = batch["image_files"]
    assert len(images)==len(condition_images)==16
    traindata_list = []
    for image, condition_image, image_path in zip(images, condition_images, image_paths):
        image_id = os.path.basename(image_path)
        image = np.array(TF.to_pil_image(image * 0.5 + 0.5))
        condition_image = np.array(TF.to_pil_image(condition_image * 0.5 + 0.5))
        traindata_sample = cv2.addWeighted(image, 0.8, condition_image, 0.6, 0)
        cv2.putText(traindata_sample, image_id, (50,50), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
        traindata_list.append(traindata_sample)
    result = None
    for i in range(4):
        hconcat = None
        for j in range(4):
            sample = traindata_list[i*4+j]
            hconcat = cv2.hconcat([hconcat, sample]) if hconcat is not None else sample
        result = cv2.vconcat([result, hconcat]) if result is not None else hconcat

    save_path_ = os.path.join(save_path, "traindata_sample.jpg")
    Image.fromarray(result).save(save_path_)
    save_path_ = os.path.join(save_path, "traindata_sample.txt")
    with open(save_path_, 'w')as f:
        for image_path in image_paths:
            f.write(image_path+'\n')
    print(f"train data sample has saved in {save_path_}")


def resize_and_crop(image:Image.Image, kps:np.ndarray, bbox:list=None, factor=2.0, size=512):
        # 1.init
        if isinstance(image, Image.Image):
            w, h = image.size
            image = np.array(image)     # [::-1]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 2. expand according to the bbox area
        if bbox is not None:
            factor = factor
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            bb_w, bb_h = x2-x1, y2-y1
            cx = x1 + bb_w // 2
            cy = y1 + bb_h // 2
            # adaptive adjustment
            crop_size = max(bb_w, bb_h)*factor
            x1 = max(0, cx-crop_size//2)
            y1 = max(0, cy-crop_size//2)
            x2 = min(w, cx+crop_size//2)
            y2 = min(h, cy+crop_size//2)
            if x2==w:
                x1 = max(0, x2-crop_size)
            if y2==h:
                y1 = max(0, y2-crop_size)
            if x1==0:
                x2 = min(w, x1+crop_size)
            if y1==0:
                y2 = min(h, y2+crop_size)
            # cut square area
            w, h = x2-x1, y2-y1
            image = image[int(y1):int(y2), int(x1):int(x2)]
            # fix kps
            kps[:, 0] = kps[:, 0] - x1
            kps[:, 1] = kps[:, 1] - y1
        
        # 3.short side resize and crop image
        if h < w:
            new_h = size
            new_w = int(new_h * (w / h))
        else:
            new_w = size
            new_h = int(new_w * (h / w))
        resized_img = cv2.resize(image, (new_w, new_h))
        # top = (new_h - self.size) // 2
        top = 0
        left = (new_w - size) // 2

        cropped_img = resized_img[top:top+size, left:left+size]
        kps[:, 0] = (kps[:, 0] * new_w / w) - left
        kps[:, 1] = (kps[:, 1] * new_h / h) - top

        return cropped_img, kps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", type=str, nargs='+',
        default=None)
    parser.add_argument("--num_tokens", type=int, default=16)
    parser.add_argument("--ckpt_name", type=str, default='sd15_faceid_portrait.bin')
    parser.add_argument("--save_name", type=str, default='test_sampling')
    parser.add_argument("--mode", type=str, default='distance',help="Union['inference', 'distance']")
    parser.add_argument("--test_data_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    # 1.init
    os.makedirs(args.save_dir, exist_ok=True) if args.save_dir is not None else None
    source_dir = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/finetune/base-portrait"
    reset_ckpt_input = [
        f"{source_dir}/20140130-sd15-crop--V1-wo_xformer-scratch",
        f"{source_dir}/20140130-sd15-crop--V1-wo_xformer-scratch_from_step6000",
        f"{source_dir}/20140205-sd15-crop--V1-wo_xformer-scratch_from_step160000",
        f"{source_dir}/20140131-sd15-crop--V1-wo_xformer-scratch_from_step26000/",
        f"{source_dir}/20140205-sd15-crop--V1-wo_xformer-scratch_from_step190000",
    ]
    args.input_dirs = reset_ckpt_input if args.input_dirs is None else args.input_dirs
    print(f"input_dirs:{args.input_dirs}\n----------------------\n")
    test_data_paths = [os.path.join(args.test_data_dir, name) for name in os.listdir(args.test_data_dir) if name.split('.')[1] != 'txt'] \
        if args.test_data_dir is not None else test_data_paths
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
    print(f"**check:{checkpoint_dirs[:5]}\n----------------------\n")
    # exit(0)
    def extract_number(input):
        dir_name = os.path.basename(os.path.dirname(input))
        pretrain_step = int(dir_name.split('step')[-1]) if 'step' in dir_name else 0
        fineturn_step = int(os.path.basename(input).split('-')[-1])
        return pretrain_step+fineturn_step

    checkpoint_dirs = sorted(checkpoint_dirs, key=extract_number)
    for checkpoint_dir in checkpoint_dirs:
        print(checkpoint_dir)

    if args.mode == 'inference':
        inference(checkpoint_dirs, args.ckpt_name, output_dir=args.save_dir)
    elif args.mode == 'distance':
        distance(checkpoint_dirs)
    elif args.mode == 'portrait_ti':
        inference_ti_token(checkpoint_dirs[0], args.ckpt_name, output_dir=args.save_dir)
    elif args.mode == 'stylegan':
        inference_styleGAN(checkpoint_dirs[0], 'sd15_faceid_wplus.bin')
    elif args.mode == 'instantid':
        inference_instantid(checkpoint_dirs[0], 'sd15_instantid.bin', output_dir=args.save_dir)
    else:
        ValueError("The mode param must be selected between inference and distance")
