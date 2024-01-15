import argparse
import copy
import functools
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
import cv2
from diffusers.loaders import LoraLoaderMixin
import transformers

from pathlib import Path
import json
import itertools
import time
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from packaging import version
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)
import logging

# JIAHUI'S MODIFY
# for ipadapter
from ip_adapter.ip_adapter_faceid import MLPProjModel
from ip_adapter.attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor
# for lora

# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, tokenizer_2, size=1024, center_crop=True, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.center_crop = center_crop
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path
    
        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        text = item["text"]
        image_file = item["image_file"]

        # JIAHUI'S MODIFY
        try:
            faces = item["faces"]
            assert len(faces) == 1
            # get metadata
            raw_image = Image.open(image_file)
            face_id_embed = torch.from_numpy(np.array(faces[0]['normed_embedding']))
            
            # original size
            original_width, original_height = raw_image.size
            original_size = torch.tensor([original_height, original_width])

            image_tensor = self.transform(raw_image.convert("RGB"))
        except OSError as e:
            print(f'An OSError occurred:{e} ==> {image_file}')
            return {
            "image": None,
            "text_input_ids": None,
            "text_input_ids_2": None,
            "face_id_embed": None,
            # "clip_image": clip_image,
            "drop_image_embed": None,   # bool
            "original_size": None,
            "crop_coords_top_left": None,
            "target_size": None,
            # 'img_path': None,
        }

         # # read image
        # raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        
        # # original size
        # original_width, original_height = raw_image.size
        # original_size = torch.tensor([original_height, original_width])
        
        # image_tensor = self.transform(raw_image.convert("RGB"))
        # +++++++++++++++++++++++++++++++++++++++++

        # random crop
        delta_h = image_tensor.shape[1] - self.size
        delta_w = image_tensor.shape[2] - self.size
        assert not all([delta_h, delta_w])
        
        if self.center_crop:
            top = delta_h // 2
            left = delta_w // 2
        else:
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        image = transforms.functional.crop(
            image_tensor, top=top, left=left, height=self.size, width=self.size
        )
        crop_coords_top_left = torch.tensor([top, left]) 

        # clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        
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
        if drop_image_embed:    # JIAHUI'S MODIFY
            face_id_embed = torch.zeros_like(face_id_embed)

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "face_id_embed": face_id_embed,     # JIAHUI'S MODIFY
            # "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,   # bool
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([self.size, self.size]),
            # 'img_path': image_file,
        }
        
    
    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data if example["image"] is not None] )
    text_input_ids = torch.cat([example["text_input_ids"] for example in data if example["text_input_ids"] is not None], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data if example["text_input_ids_2"] is not None], dim=0)
    face_id_embed = torch.stack([example["face_id_embed"] for example in data if example["face_id_embed"] is not None], dim=0)
    # clip_images = torch.cat([example["clip_image"] for example in data if example["clip_image"] is not None], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data if example["drop_image_embed"] is not None]
    original_size = torch.stack([example["original_size"] for example in data if example["original_size"] is not None])
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data if example["crop_coords_top_left"] is not None])
    target_size = torch.stack([example["target_size"] for example in data if example["target_size"] is not None])
    # image_file = [example['img_path'] for example in data if example["img_path"] is not None]

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "face_id_embed": face_id_embed,   
        # "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
        # 'img_path': image_file,
    }

    

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None,
                 only_load_adapter=False,
                 ):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        logger.info(f"using pre-trained models: {ckpt_path}")   # qirui's modify
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path) if only_load_adapter is False else self.only_load_for_adapter(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs).sample
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
        # required=True,    # JIAHUI'S MODIFY
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        # required=True,    # JIAHUI'S MODIFY
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
        default=1024,
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
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
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
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint and are suitable for resuming training"
            " using `--resume_from_checkpoint`."
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

    # JIAHUI'S MODIFY
    parser.add_argument("--only_load_adapter", type=bool, default=False)
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=128)
    
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    # adding additional parameters for IP-adapter training
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument('--optimizer', default = 'adamw', type = str)
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument('--enable_xformers_memory_efficient_attention', action = 'store_true')
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    # 1. init
    args = parse_args()
    # qirui's modify
    args.output_dir = "{}_optim_{}_bs_{}_lr_{}".format(args.output_dir, args.optimizer, args.train_batch_size, args.learning_rate)
    if args.pretrained_ip_adapter_path is not None:
        args.output_dir ="{}_Pretrained".format(args.output_dir)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir,'commandline_args.txt'), 'w') as f:
        f.write(str(vars(args)))
    logging_dir = Path(args.output_dir, args.logging_dir)
    if os.path.isdir(logging_dir) is not None:
        os.makedirs(logging_dir, exist_ok=True)
    print(logging_dir)
    # +++++++++++++++

    project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    # logging_dir = Path(args.output_dir, args.logging_dir)

    # accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,       # qirui's modify
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )

    if accelerator.is_main_process:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=True)
        import transformers
        transformers.utils.logging.set_verbosity_info()
        

    # 2. Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    # vae = AutoencoderKL.from_pretrained("/mnt/nfs/file_server/public/qirui/ckpt/models--madebyollin--sdxl-vae-fp16-fix/snapshots/97b09ba005d991b0aa24996f1457909b97482caf/", subfolder="vae")
    vae = AutoencoderKL.from_pretrained(
        "/mnt/nfs/file_server/public/qirui/ckpt/models--madebyollin--sdxl-vae-fp16-fix/snapshots/97b09ba005d991b0aa24996f1457909b97482caf/",
    )
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    # image_encoder.requires_grad_(False)

    # qirui’s modify
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    # ++++++++

    # 3. set ip-adapter
    # JIAHUI'S MODIFY
    num_tokens = args.num_tokens
    lora_rank = args.lora_rank
    image_proj_model = MLPProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        id_embeddings_dim=512,
        num_tokens=num_tokens,
    )

    # init adapter modules
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
        
        # JIAHUI'S MODIFY
        if cross_attention_dim is None:
            attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank,
            )
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = LoRAIPAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0, rank=lora_rank, num_tokens=num_tokens,
            )
            # Since face plus does not have lora parameters, the parameters of miss are randomly initialized
            attn_procs[name].load_state_dict(weights, strict=False)

            
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path, args.only_load_adapter)
    
    # 4. for mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)# use fp32            # The VAE is in float32 to avoid NaN losses.
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    # image_encoder.to(accelerator.device, dtype=weight_dtype)

    # qirui's modify
    counter = 0
    name_list = [] 
    for name, params in ip_adapter.named_parameters():
        if params.requires_grad==True:
            name_list.append(name)
    print( len(name_list))
    print(len(list(ip_adapter.image_proj_model.parameters())))
    print(len(list(ip_adapter.adapter_modules.parameters())))
    # ++++++++++++++++++++++++++++++
            
            
        
    # 6. optimizer
    # JIAHUI'S MODIFY
    params_to_opt = itertools.chain(
        ip_adapter.image_proj_model.parameters(),  
        ip_adapter.adapter_modules.parameters(),
    )

    # qirui's modify
    # params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(),  ip_adapter.adapter_modules.parameters())
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
        logger.info("***** Using ADAMW8bit *****")
    elif args.optimizer == 'CAME':
        from came_pytorch import CAME
        optimizer_class = CAME
        logger.info("***** Using CAME *****")
    else:
        import transformers
        optimizer_class = transformers.AdamW
        logger.info("***** Using AdamW *****")
        
    if args.optimizer == 'CAME':
        # Qirui modified for CAME datasets
        optimizer = optimizer_class(
            params_to_opt,
            lr=args.learning_rate,
            betas=(0.9,0.999,0.9999),
            weight_decay=1e-2,
            eps=(1e-30, 1e-16),
        )
    else:
        optimizer = optimizer_class(
            params_to_opt,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    # +++++++++++++++++++

    # 7. dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, image_root_path=args.data_root_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # qirui's modify
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
 
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    # +++++++++++++++++++
        
    # 8. Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)
    
    # qirui's modify
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    print(num_update_steps_per_epoch)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("IP-adapter", config=vars(args))
    # +++++++++++++++
    
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f" Num examples = {len(train_dataset)}")
    logger.info(f" Num Epochs = {args.num_train_epochs}")
    logger.info(f" Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f" Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f" Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f" Total optimization steps = {args.max_train_steps}") 
    
    # 9. Training
    # qirui's mdify
    global_step = 0
    first_epoch = 0
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    # ++++++++++++++++
    for epoch in range(first_epoch, args.num_train_epochs):
        # ip_adapter.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(batch["images"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                # JIAHUI'S MODIFY
                image_embeds = batch["face_id_embed"].to(accelerator.device, dtype=weight_dtype)
            
                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) # concat
                        
                # add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                
                noise_pred = ip_adapter(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, image_embeds)
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(params_to_opt, args.max_grad_norm)
                # Backpropagate
                optimizer.step()
                optimizer.zero_grad()

            # qirui's modify
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                    ##################### Qirui Modification ################################
                    # if global_step % args.validation_steps == 0 or global_step == 1:
                    #     image_logs = log_validation(
                    #         vae, unet, controlnet, args, accelerator, weight_dtype, global_step, test_dataset
                    #     )
                    #####################################################
            logs = {"loss": avg_loss, "lr": args.learning_rate}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
            # ++++++++++++++++++
        accelerator.wait_for_everyone()
                
if __name__ == "__main__":
    main()   
