import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import wandb

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
import accelerate
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from torchvision.transforms.functional import to_pil_image
import numpy as np

from ip_adapter.ip_adapter_faceid import MLPProjModel
from ip_adapter.resampler import Resampler
from ip_adapter.utils import is_torch2_available
# from ip_adapter.attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

from accelerate.logging import get_logger
logger = get_logger(__name__)
import logging
from my_script.deepface.eval_model import distance, inference, inference_ti_token
import subprocess

# Dataset
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "id_embed_file": "faceid.bin"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        text = item["text"]
        image_file = item["image_file"]
        embeds_path = item["embeds_path"]
        
        try:
            raw_image = Image.open(image_file)
            image = self.transform(raw_image.convert("RGB"))
            face_id_embed = torch.from_numpy(np.load(embeds_path))
            clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        except Exception as e:
            print(e)
            return {
            "image": None,
            "text_input_ids": None,
            "face_id_embed": None,
            "drop_image_embed": None,
            "clip_image": None,
        }
        
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
            clip_image = torch.zeros_like(clip_image)
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
            "text_input_ids": text_input_ids,
            "face_id_embed": face_id_embed,
            "drop_image_embed": drop_image_embed,
            "clip_image": clip_image,
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data if example["image"] is not None])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data if example["text_input_ids"] is not None], dim=0)
    face_id_embed = torch.stack([example["face_id_embed"] for example in data if example["face_id_embed"] is not None])
    clip_images = torch.cat([example["clip_image"] for example in data if example["clip_image"] is not None], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data if example["drop_image_embed"] is not None]

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "face_id_embed": face_id_embed,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
    }
    

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, text_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.text_proj_model = text_proj_model
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, ip_image_embeds, ip_text_embeds):
        ip_text_tokens = self.text_proj_model(ip_text_embeds)
        ip_image_tokens = self.image_proj_model(ip_image_embeds)
        encoder_hidden_states = torch.cat([ip_text_tokens, encoder_hidden_states, ip_image_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_image_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_text_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.text_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.text_proj_model.load_state_dict(state_dict["text_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_image_proj_model_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_text_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.text_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_image_proj_sum != new_image_proj_model_sum, "Weights of image_proj_model did not change!"
        assert orig_text_proj_sum != new_text_proj_sum, "Weights of image_proj_model did not change!"
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
    parser.add_argument("--ti_num_tokens", type=int, required=True)
    parser.add_argument("--deepface_run_step", type=int, required=True)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--discriminator_learning_rate", type=float, default=3e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lantent_type", type=str, default="prev")
    # deepspeed
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    # +++++++++++++++++++++++++++++++++++
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    # 1.init
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        # save param
        import sys
        param_save_path = os.path.join(args.output_dir, "param.txt")
        with open(param_save_path, 'w')as f:
            for param in sys.argv:
                f.write(param+'\n')
        # init wandb
        if args.report_to == "wandb":
            print("init wandb trackers")
            accelerator.init_trackers("portrait-ti_token-id_discriminator", 
                config=dict(vars(args)),
                init_kwargs={
                    "wandb": {"name": os.path.basename(args.output_dir),}
                },
            )


    # 2.Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)


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
    

    # 4. set mixed precision
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    

    # 5. Initializes learnable modules
    # 5.1 Discriminator
    from my_script.module.loss import IdentityDiscriminator
    id_discriminator = IdentityDiscriminator(
        disc_start=0, 
        disc_in_channels=4, 
        disc_num_layers=3
        )        # modify for discriminator
    # 5.2 Ip-adapter
    image_proj_model = MLPProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        id_embeddings_dim=512,
        num_tokens=args.num_tokens,
    )
    text_proj_model = MLPProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        id_embeddings_dim=image_encoder.config.projection_dim,
        num_tokens=args.ti_num_tokens,
    )

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
    ip_adapter = IPAdapter(unet, image_proj_model, text_proj_model, adapter_modules, args.pretrained_ip_adapter_path)

    # check param
    if accelerator.is_main_process:
        name_list = [] 
        for name, params in ip_adapter.named_parameters():
            if params.requires_grad==True:
                name_list.append(name)
        for name, params in id_discriminator.named_parameters():
            if params.requires_grad==True:
                name_list.append(name)
        print( len(name_list))
        print(len(list(ip_adapter.image_proj_model.parameters())))
        print(len(list(ip_adapter.text_proj_model.parameters())))
        print(len(list(ip_adapter.adapter_modules.parameters())))
        print(len(list(id_discriminator.discriminator.parameters())))
    

    # 6. optimizer
    G_params_to_opt = itertools.chain(
        ip_adapter.image_proj_model.parameters(),  
        ip_adapter.adapter_modules.parameters(),
        ip_adapter.text_proj_model.parameters(),
        )
    D_params_to_opt = id_discriminator.discriminator.parameters()
    
    # ## use deepspeed
    # accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"] = {
    #     "type": "AdamW",
    #     "params": {
    #         "lr": args.learning_rate,
    #         "betas": [args.adam_beta1, args.adam_beta2],
    #         "eps": args.adam_epsilon,
    #         "weight_decay": args.adam_weight_decay,
    #     },
    # }
    # optimizer_class = accelerate.utils.DummyOptim
    # G_optimizer = optimizer_class(G_params_to_opt)
    # accelerator.state.deepspeed_plugin.deepspeed_config["optimizer"] = {
    #     "type": "AdamW",
    #     "params": {
    #         "lr": args.discriminator_learning_rate,
    #         "betas": [args.adam_beta1, args.adam_beta2],
    #         "eps": args.adam_epsilon,
    #         "weight_decay": args.adam_weight_decay,
    #     },
    # }
    # D_optimizer = optimizer_class(D_params_to_opt)

    # unused deepspeed
    G_optimizer = torch.optim.AdamW(G_params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    D_optimizer = torch.optim.AdamW(D_params_to_opt, lr=args.discriminator_learning_rate, weight_decay=args.weight_decay)


    # 7. dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution, image_root_path=args.data_root_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    

    # 8. Scheduler and math around the number of training steps.
    import math
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps


    # 9. Prepare everything with our `accelerator`.
    # modify for discriminator
    ip_adapter, id_discriminator, G_optimizer, D_optimizer, train_dataloader = \
        accelerator.prepare(ip_adapter, id_discriminator, G_optimizer, D_optimizer, train_dataloader)


    # 10. log info
    if accelerator.is_main_process:
        # Afterwards we recalculate our number of training epochs
        total_batch_size = args.train_batch_size * accelerator.num_processes
        logger.info("***** Running training *****")
        logger.info(f" Num examples = {len(train_dataset)}")
        logger.info(f" Num Epochs = {args.num_train_epochs}")
        logger.info(f" Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f" Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f" Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f" Total optimization steps = {args.max_train_steps}") 
        logger.info(f" Output Dir = {args.output_dir}") 
        logger.info(f" XFormer Enable = {args.enable_xformers_memory_efficient_attention}") 

    # torch.backends.cuda.enable_flash_sdp(False)
    

    # 11. training
        
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
                
                    ip_image_embeds = batch["face_id_embed"].to(accelerator.device, dtype=weight_dtype)
                    clip_images = batch["clip_images"].to(accelerator.device, dtype=weight_dtype)
                
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
                        # ip_text_embeds = image_encoder(clip_images, output_hidden_states=True).hidden_states[-2]
                        ip_text_embeds = image_encoder(clip_images).image_embeds
                    
                    noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, ip_image_embeds, ip_text_embeds)


                    # modify for discriminator
                    ## ori
                    # loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                    # avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                    # accelerator.backward(loss)
                    # optimizer.step()
                    # optimizer.zero_grad()
                    # if accelerator.is_main_process:
                    #     print("Epoch {}, global_step {}, data_time: {}, time: {}, step_loss: {}".format(
                    #         epoch, global_step, load_data_time, time.perf_counter() - begin, avg_loss))

                    ## my method
                    G_optimizer.zero_grad()
                    g_loss, log0, debug_image = id_discriminator(
                        noise, noise_pred, noisy_latents, 
                        timesteps, 
                        noise_scheduler, 
                        optimizer_idx=0,
                        global_step=global_step,
                        lantent_type=args.lantent_type,
                        )
                    accelerator.backward(g_loss)
                    G_optimizer.step()

                    D_optimizer.zero_grad()
                    d_loss, log1, _ = id_discriminator(
                        noise, noise_pred, noisy_latents, 
                        timesteps, 
                        noise_scheduler, 
                        optimizer_idx=1,
                        global_step=global_step,
                        lantent_type=args.lantent_type,
                        )
                    accelerator.backward(d_loss)
                    D_optimizer.step()
                    if accelerator.is_main_process and global_step%args.print_freq == 0:
                        print("Epoch {}, global_step {}, data_time: {}, time: {}".format(
                            epoch, global_step, load_data_time, time.perf_counter() - begin))
                        accelerator.log({**log0, **log1}, step=global_step)
                        debug_image['ori_image'] = to_pil_image(batch["images"][-1] * 0.5 + 0.5).resize((256,256))
                        image_log = []
                        t = debug_image.pop('t')
                        for k, v in debug_image.items():
                            image_log.append(wandb.Image(v, caption=k+'--'+str(t)))
                        accelerator.log({'image_log':image_log}, step=global_step)
                    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            global_step += 1
            
            accelerator.wait_for_everyone()
            if global_step % args.save_steps == 0 and accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                # accelerator.save_state(save_path)
                # jiahui's modify
                os.makedirs(save_path, exist_ok=True)
                sd = accelerator.unwrap_model(ip_adapter).state_dict()
                result = {
                    'image_proj':{},
                    'text_proj':{},
                    'ip_adapter':{},
                }
                for k in sd:
                    if k.startswith("unet"):
                        pass
                    elif k.startswith("image_proj_model"):
                        result['image_proj'][k.replace("image_proj_model.", "")] = sd[k]
                    elif k.startswith("adapter_modules"):
                        result['ip_adapter'][k.replace("adapter_modules.", "")] = sd[k]
                    elif k.startswith("text_proj_model"):
                        result['text_proj'][k.replace("text_proj_model.", "")] = sd[k]
                save_path_ = os.path.join(save_path, 'sd15_faceid_portrait.bin')
                logger.info(f"saving ckpt file ==> {save_path_}")
                accelerator.save(result, save_path_)
                print(f"ckpt has saved in {save_path_}")
                inference_ti_token(save_path, 'sd15_faceid_portrait.bin')

                # if global_step % args.deepface_run_step == 0:
                #     logger.info("running deepface for model eval")
                #     subprocess.Popen([
                #         "/home/mingjiahui/anaconda3/envs/ipadapter/bin/python", 
                #         "./my_script/deepface/eval_model.py", 
                #         "--mode", 
                #         "distance",
                #         "--input_dirs",
                #         f"{save_path}",
                #         ])

                # ++++++++++++++++++++++++++++++++++q
            
            begin = time.perf_counter()

            # print("debug")
            # exit(0)
                
if __name__ == "__main__":
    main()    
