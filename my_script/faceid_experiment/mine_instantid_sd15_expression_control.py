from torchvision import transforms
import torch
import os
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import sys

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
from my_script.deepface.eval_model import resize_and_crop


def inference_instantid(checkpoint_dir, ckpt_name, resampler=True, num_tokens=16, output_dir=None):
    # 1. init
    test_image_path = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data/guonan.jpg"
    prompt_template= "a young woman with a {}, wearing a pink shirt. She is standing in front of a fence, possibly in a park or an outdoor setting. The woman appears to be enjoying her time outdoors, possibly engaging in a sport or a recreational activity. "
    expression_keys = ['bright smile', 'sad face', 'astonished face', 'exaggerated expression']
    output_dir = os.path.join(checkpoint_dir, 'test_sampling') if output_dir is None else output_dir
    instantid_weights = [round(i, 1) for i in np.arange(0, 1+0.2, 0.2).tolist()]

    os.makedirs(output_dir, exist_ok=True)

    # 2. init face model
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
    

    # 3. get face info
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
    ])
    face_image = Image.open(test_image_path).convert("RGB")

    face_image = transform(face_image)
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    assert len(face_info) > 0, f"no face find ==> {test_image_path}"
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]   # only use the maximum face
    face_emb = torch.from_numpy(face_info.normed_embedding).unsqueeze(0).unsqueeze(0)

    cropped_img, kps = resize_and_crop(face_image, face_info['kps'], face_info['bbox'].tolist(), factor=2)
    cropped_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    face_kps = draw_kps(cropped_img, kps)
    

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

    

    # 4.4 load ip-adapter
    ip_ckpt = os.path.join(checkpoint_dir, ckpt_name)
    ip_model = InstantIDFaceID(pipe, ip_ckpt, device, num_tokens=num_tokens, n_cond=1, resampler=resampler)

    # 4.5 generate image
    for expression_key in tqdm(expression_keys):
        # prompt
        prompt = prompt_template.format(expression_key)
        result = None
        for instantid_weight in instantid_weights:
            # processing
            image = ip_model.generate(
                prompt=prompt,
                num_samples=1, 
                width=512, height=512, 
                num_inference_steps=30, 
                seed=42, 
                guidance_scale=6,
                faceid_embeds=face_emb, image=face_kps,
                # control weight
                scale=instantid_weight,
                controlnet_conditioning_scale=instantid_weight,

            )[0]
            result = cv2.hconcat([result, np.array(image)]) if result is not None else np.array(image)

        # save
        save_name = os.path.basename(test_image_path)
        prefix, suffix = save_name.split('.')
        save_path = os.path.join(output_dir, prefix+expression_key.replace(' ', '_')+'.'+suffix)
        Image.fromarray(result).save(save_path)
        print(f"image result has saved in {save_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    inference_instantid(args.ckpt, 'sd15_instantid.bin', output_dir=args.save_dir)
