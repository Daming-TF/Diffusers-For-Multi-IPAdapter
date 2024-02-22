import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
from PIL import Image
import numpy as np
import cv2
from typing import Union

import torch
import torch.nn.functional as F
from torchvision import transforms

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel
from diffusers.utils.torch_utils import is_compiled_module
import transformers
from transformers import CLIPImageProcessor
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from ip_adapter.ip_adapter_faceid import MLPProjModel
from ip_adapter.utils import is_torch2_available
# from ip_adapter.attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

# jiahui's modify
from accelerate.logging import get_logger
logger = get_logger(__name__)
import logging
import subprocess
from my_script.deepface.eval_model import distance, inference, inference_instantid, save_sample_pic
from InstantID.pipeline_stable_diffusion_xl_instantid import draw_kps
from InstantID.ip_adapter.resampler import Resampler
# +++++++++++++++++

# Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path="", factor=2):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
        self.factor = factor

        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "id_embed_file": "faceid.bin"}]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.conditioning_transform = transforms.ToTensor()
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        text = item["text"]
        image_file = item["image_file"]
        embeds_path = item["embeds_path"]
        json_path = item["face_info_json"]
        
        # JIAHUI'S MODIFY
        try:
            raw_image = Image.open(image_file).convert("RGB")
            face_id_embed = torch.from_numpy(np.load(embeds_path))
            with open(json_path, 'r')as f:
                face_info = json.load(f)
                kps = face_info['kps'][0]
                bbox = face_info['bbox'][0]
            
            cropped_img, kps = self.resize_and_crop(raw_image, np.array(kps), bbox)
            # cropped_img = Image.fromarray(cropped_img[::-1])
            cropped_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
            image = self.transform(cropped_img)
            face_kps = draw_kps(cropped_img, kps)
            face_kps = self.conditioning_transform(face_kps)
        except Exception as e:
            print(e)
            return {
            "image": None,
            "condition_image": None,
            "text_input_ids": None,
            "face_id_embed": None,
            "drop_image_embed": None,
            "image_path": None,
        }

        # # read image
        # raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        # image = self.transform(raw_image.convert("RGB"))

        # face_id_embed = torch.load(item["id_embed_file"], map_location="cpu")
        # face_id_embed = torch.from_numpy(face_id_embed)
        # ++++++++++++++++++++++++++++++++++
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        if drop_image_embed:
            face_id_embed = torch.zeros_like(face_id_embed)
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "condition_image": face_kps,
            "text_input_ids": text_input_ids,
            "face_id_embed": face_id_embed,
            "drop_image_embed": drop_image_embed,
            "image_file": image_file,
        }

    def __len__(self):
        return len(self.data)
    
    def resize_and_crop(self, image:Image.Image, kps:np.ndarray, bbox:Union[None, list]=None):
        # 1.init
        if isinstance(image, Image.Image):
            w, h = image.size
            image = np.array(image)     # [::-1]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 2. expand according to the bbox area
        if bbox is not None:
            factor = self.factor
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
        
        # 3.short side resize
        if h < w:
            new_h = self.size
            new_w = int(new_h * (w / h))
        else:
            new_w = self.size
            new_h = int(new_w * (h / w))
        resized_img = cv2.resize(image, (new_w, new_h))
        # top = (new_h - self.size) // 2
        top = 0
        left = (new_w - self.size) // 2

        cropped_img = resized_img[top:top+self.size, left:left+self.size]
        kps[:, 0] = (kps[:, 0] * new_w / w) - left
        kps[:, 1] = (kps[:, 1] * new_h / h) - top

        return cropped_img, kps

def collate_fn(data):
    images = torch.stack([example["image"] for example in data if example["image"] is not None])
    condition_images = torch.stack([example["condition_image"] for example in data if example["condition_image"] is not None])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data if example["text_input_ids"] is not None], dim=0)
    face_id_embeds = torch.stack([example["face_id_embed"] for example in data if example["face_id_embed"] is not None])
    drop_image_embeds = [example["drop_image_embed"] for example in data if example["drop_image_embed"] is not None]
    image_files = [example["image_file"] for example in data if example["image_file"] is not None]

    return {
        "images": images,
        "condition_images": condition_images,
        "text_input_ids": text_input_ids,
        "face_id_embeds": face_id_embeds,
        "drop_image_embeds": drop_image_embeds,
        "image_files": image_files,
    }
    

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path, controlnet, dtype):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.controlnet = controlnet
        self.weight_dtype = dtype

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, face_id_embeds, condition_images):
        ip_tokens = self.image_proj_model(face_id_embeds)   # {b, num_token, 768}
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)    # {b, num_token+77, 768}
        # controlnet
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=ip_tokens,
            controlnet_cond=condition_images,
            return_dict=False,
        )
        # Predict the noise residual
        noise_pred = self.unet(
            noisy_latents, 
            timesteps, 
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=[
                        sample.to(dtype=self.weight_dtype) for sample in down_block_res_samples
                    ],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=self.weight_dtype),
            ).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        required=True,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        # required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        # required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # jiahui's modify
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action='store_true')
    parser.add_argument("--num_tokens", type=int, required=True)
    parser.add_argument("--deepface_run_step", type=int, default=None)
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--factor", type=int, default=2)
    # +++++++++++++++++++++++++++++++++
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    # 1. init
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # from accelerate import DistributedDataParallelKwargs
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # Make one log on every process with the configuration for debugging.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    if args.controlnet_model_name_or_path is not None:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet)

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    # 3. init xformer
    if args.enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            import xformers
            from packaging import version
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    # 9. set mixed precision
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    # controlnet.to(accelerator.device, dtype=weight_dtype)
    #image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # 4. init ip-adapter
    image_proj_model = Resampler(
            dim=unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,       # sd15 12   sdxl 20
            num_queries=args.num_tokens,
            embedding_dim=512,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4,
        )

    # init adapter modules
    # lora_rank = 128
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            # attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            # attn_procs[name] = LoRAIPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=args.num_tokens)
            attn_procs[name].load_state_dict(weights, strict=False)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())   
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path, controlnet, dtype=weight_dtype)

    # check param
    name_list = [] 
    for name, params in ip_adapter.named_parameters():
        if params.requires_grad==True:
            name_list.append(name)
    print(len(name_list))
    print(len(list(ip_adapter.image_proj_model.parameters())))
    print(len(list(ip_adapter.adapter_modules.parameters())))
    print(len(list(ip_adapter.controlnet.parameters())))
    
    # 5. optimizer
    params_to_opt = itertools.chain(
        ip_adapter.image_proj_model.parameters(),  
        ip_adapter.adapter_modules.parameters(),
        ip_adapter.controlnet.parameters(),
        )
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 6. dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution, image_root_path=args.data_root_path, factor=args.factor)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # 7. Scheduler and math around the number of training steps.
    import math
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # 8. Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)

    # 11. log info
    if accelerator.is_main_process:
        # Afterwards we recalculate our number of training epochs
        logger.info("***** Running training *****")
        logger.info(f" Num examples = {len(train_dataset)}")
        logger.info(f" Num Epochs = {args.num_train_epochs}")
        logger.info(f" Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f" Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        # logger.info(f" Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f" Total optimization steps = {args.max_train_steps}") 
        logger.info(f" Output Dir = {args.output_dir}") 
        logger.info(f" XFormer Enable = {args.enable_xformers_memory_efficient_attention}") 
        logger.info(f" Data Factor = {args.factor}") 
    # torch.backends.cuda.enable_flash_sdp(False)
    
    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(ip_adapter):
                with torch.backends.cuda.sdp_kernel(enable_flash=False) as disable:     # jiahui's mpdify
                    # Convert images to latent space
                    with torch.no_grad():
                        latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                    face_id_embeds = batch["face_id_embeds"].to(accelerator.device, dtype=weight_dtype)
                    condition_images = batch["condition_images"].to(accelerator.device, dtype=weight_dtype)
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
                    
                    noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, face_id_embeds, condition_images)
            
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                
                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                    
                    # Backpropagate
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                    if accelerator.is_main_process:
                        print("Epoch {}, global_step {}, data_time: {}, time: {}, step_loss: {}".format(
                            epoch, global_step, load_data_time, time.perf_counter() - begin, avg_loss))
            
            global_step += 1
            
            accelerator.wait_for_everyone()
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                # accelerator.save_state(save_path)
                os.makedirs(save_path, exist_ok=True)
                ip_model = accelerator.unwrap_model(ip_adapter)
                sd = ip_model.state_dict()
                result_0 = {
                    'image_proj':{},
                    'ip_adapter':{},
                }
                # result_1 = {}
                for k in sd:
                    if k.startswith("unet"):
                        pass
                    elif k.startswith("image_proj_model"):
                        result_0['image_proj'][k.replace("image_proj_model.", "")] = sd[k]
                    elif k.startswith("adapter_modules"):
                        result_0['ip_adapter'][k.replace("adapter_modules.", "")] = sd[k]
                    # elif k.startswith('controlnet'):
                    #     result_1[k.replace("controlnet.", "")] = sd[k]
                
                save_path_ = os.path.join(save_path, 'sd15_instantid.bin')
                accelerator.save(result_0, save_path_)
                # save_path_ = os.path.join(save_path, ,'diffusion_pytorch_model.bin')
                # accelerator.save(result_1, save_path_)
                print(f"ckpt has saved in {save_path_}")

                def unwrap_model(model):
                    model = accelerator.unwrap_model(model)
                    model = model._orig_mod if is_compiled_module(model) else model
                    return model
                ip_model.controlnet = unwrap_model(ip_model.controlnet)
                ip_model.controlnet.save_pretrained(save_path)
                save_sample_pic(save_path, batch)
                inference_instantid(save_path, 'sd15_instantid.bin')
                # distance(save_path)
                if args.deepface_run_step is not None and global_step % args.deepface_run_step == 0:
                    subprocess.Popen([
                        "/home/mingjiahui/anaconda3/envs/ipadapter/bin/python", 
                        "./my_script/deepface/eval_model.py", 
                        "--mode", 
                        "distance",
                        "--input_dirs",
                        f"{save_path}",
                        ])
            
            begin = time.perf_counter()
                
if __name__ == "__main__":
    main()    
