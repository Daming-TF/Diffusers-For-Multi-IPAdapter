import torch
import os
import argparse


def transfer_ckpt(ckpt_dir, output_name='sdxl_faceid.bin'):
    print("transfer the ckpt......")
    sd = torch.load(os.path.join(ckpt_dir, 'pytorch_model.bin'), map_location="cpu")
    image_proj_sd = {}
    ip_sd = {}
    for k in sd:
        if k.startswith("unet"):
            pass
        elif k.startswith("image_proj_model"):
            image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
        elif k.startswith("adapter_modules"):
            ip_sd[k.replace("adapter_modules.", "")] = sd[k]
    # save
    target_ckpt = os.path.join(ckpt_dir, output_name)
    torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, target_ckpt)
    return target_ckpt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_name", type=str, default='sdxl_faceid.bin')
    args = parser.parse_args()
    transfer_ckpt(args.input_dir, args.output_name)