# Salesforce/blip-image-captioning-base
import os
from huggingface_hub import snapshot_download
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--vision',
    type=str,
    required=True,
)
parser.add_argument(
    '--save_dir',
    type=str,
    default=r'f:\seekoo\models',
)
parser.add_argument(
    '--allow_patterns',
    type=str,
    nargs='+',
    default=None
)
parser.add_argument(
    '--ignore_patterns',
    type=str,
    nargs='+',
    default=None
)
args = parser.parse_args()
vision = args.vision
save_name = vision.replace('/', '--')
save_dir = r"/mnt/nfs/file_server/public/mingjiahui/models/" if args.save_dir is None else args.save_dir
save_path = os.path.join(save_dir, save_name)
print(f'Downing...... ==> {args.vision} To {save_path}')
snapshot_download(
    repo_id=vision,
    local_dir=save_path,
    allow_patterns=args.allow_patterns,
    ignore_patterns=args.ignore_patterns
)


# from huggingface_hub import hf_hub_download
# photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")