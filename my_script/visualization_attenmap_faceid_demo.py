import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
import copy

from insightface.app import FaceAnalysis
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.utils import face_align
from numpy.linalg import norm as l2norm
import cv2

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDPlus, IPAdapterFaceID
from ip_adapter.utils import register_cross_attention_hook, get_net_attn_map, attnmaps2images

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

v2 = False
source_dir = r'/mnt/nfs/file_server/public/mingjiahui/models'
base_model_path =f"{source_dir}/SG161222--Realistic_Vision_V4.0_noVAE/"   # "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = f"{source_dir}/stabilityai--sd-vae-ft-mse/"   # "stabilityai/sd-vae-ft-mse"
image_encoder_path = fr"{source_dir}/h94--IP-Adapter/h94--IP-Adapter/models/image_encoder/"
ip_ckpt = fr"{source_dir}/h94--IP-Adapter/faceid/ip-adapter-faceid-plusv2_sd15.bin"
device = "cuda"

print("loading models......")
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
print("resister hook......")
pipe.unet = register_cross_attention_hook(pipe.unet)


# generate image
prompt = "photo of a woman in red dress in a garden, white hair, happy"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

import wandb
table = wandb.Table(columns=["prompt", "scale", "face", "gen"])

def rtn_face_get(self, img, face):
    aimg = face_align.norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
    #print(cv2.imwrite("aimg.png", aimg))
    face.embedding = self.get_feat(aimg).flatten()
    face.crop_face = aimg
    return face.embedding

print("getting face embeds......")
ArcFaceONNX.get = rtn_face_get
image = cv2.imread("/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data/aoteman.jpg")
faces = app.get(image)
faceid_embeds = faces[0].normed_embedding
faceid_embeds = torch.from_numpy(faceid_embeds).unsqueeze(0)
face_image = faces[0].crop_face

print("create ip model......")
plus_ip_model = IPAdapterFaceIDPlus(copy.deepcopy(pipe), image_encoder_path, ip_ckpt, device)
print("processing......")
images = plus_ip_model.generate(
    prompt=prompt,
    negative_prompt=negative_prompt,
    face_image=face_image,
    faceid_embeds=faceid_embeds,
    shortcut=v2,
    s_scale=1,
    num_samples=1,
    width=512, height=768,
    num_inference_steps=30, seed=2023
)

attn_maps = get_net_attn_map((768, 512))
print(attn_maps.shape)
attn_hot = attnmaps2images(attn_maps)

import matplotlib.pyplot as plt
#axes[0].imshow(attn_hot[0], cmap='gray')
display_images = [cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)] + attn_hot + [images[0]]
fig, axes = plt.subplots(1, len(display_images), figsize=(12, 4))
for axe, image in zip(axes, display_images):
    axe.imshow(image, cmap='gray')
    axe.axis('off')
plt.savefig("./data/other/debug0.jpg")


ip_model = IPAdapterFaceID(copy.deepcopy(pipe), ip_ckpt, device)
images = ip_model.generate(
    prompt=prompt, negative_prompt=negative_prompt,
    faceid_embeds=faceid_embeds,
    num_samples=1,
    width=512, height=768,
    num_inference_steps=30, seed=2023
)

attn_maps = get_net_attn_map((768, 512))
print(attn_maps.shape)
attn_hot = attnmaps2images(attn_maps)

display_images = [cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)] + attn_hot + [images[0]]
fig, axes = plt.subplots(1, len(display_images), figsize=(12, 4))
for axe, image in zip(axes, display_images):
    axe.imshow(image, cmap='gray')
    axe.axis('off')
plt.savefig("./data/other/debug1.jpg")