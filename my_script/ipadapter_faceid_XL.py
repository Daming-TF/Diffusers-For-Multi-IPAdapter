
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL

# 1. get face embeds
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

image = cv2.imread(r"")
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

# 3. load ip-adapter
ip_model = IPAdapterFaceIDXL(pipe, ip_ckpt, device)

# 4. generate image
prompt = "A closeup shot of a beautiful Asian teenage girl in a whissh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCt9tKsq2x/acgD4Gb+FKBxAu7EZoU38wvKZecgvk0EF8PDISD+wp0rS5YNKV7U9bHIqrvyK/ji8FQn52qaDov33opf4F1SvNxUHEkf9VFsgcS2BKoJONMVIO959Qy94eJq2b6K/DZEM+h/E4RD3C34hJSX8gKNLWfvKeyM+oakivfPG5KUlH7MYU4FxGfDuzgFyrU+jQk1hXOFwkm55OSPU6FGT5rk3jNogDi6DEJLGwU18Ayors6iAYpSBn6Pqw/DFRLoYgdh/rnLOf60qJet3XEt2kx9FclA2/a2/qWCvP6xl8XvXBk5g5PsLaYvqUdsWgbWN/V8LCegRy6jdUeg7GvU7dnm47DlYOgPvhRxjDxsRN+pw8To2OaduMtlrvX+RJ8gDd+PoiR+jRpkCayinP9CQJEW40zj44bkHspbuX9BUIHTDWeVkJcd0A9V4Na650HKFtQr2to+HhTqYxDZpeeIbVELyuUnIB+E/TrEYQTp2ZkETvRqm4RyRv9xmKk= mingjiahui@pste dress wearing small silver earrings in the garden, under the soft morning light"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

images = ip_model.generate(
    prompt=prompt, negative_prompt=negative_prompt, faceid_embeds=faceid_embeds, num_samples=2,
    width=1024, height=1024,
    num_inference_steps=30, guidance_scale=7.5, seed=2023
)