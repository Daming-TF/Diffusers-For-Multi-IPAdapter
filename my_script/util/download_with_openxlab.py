
import openxlab
import argparse
import os


def main(args):
    AccessKey = '95ye0arz3ydlwggay7pw' 
    SecretKey = 'kn63lvrz4m2z9e5pqkyvenaqzwq0roanodbpj8lb'
    openxlab.login(ak=AccessKey, sk=SecretKey) #进行登录，输入对应的AK/SK

    from openxlab.dataset import info
    info(dataset_repo=args.data_name) #数据集信息查看

    from openxlab.dataset import query
    query(dataset_repo=args.data_name) #数据集文件列表查看

    from openxlab.dataset import get
    get(dataset_repo=args.data_name, target_path=args.save_dir)  # 数据集下载

    from openxlab.dataset import download
    download(dataset_repo=args.data_name,source_path='/README.md', target_path=args.save_dir) #数据集文件下载


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True) # OpenDataLab/VGGFace2
    parser.add_argument("--save_dir", type=str,required=True)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)