import argparse
import os
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
import torch
import json
import matplotlib.pyplot as plt

import sys
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_dir))

from insightface.app import FaceAnalysis

from data.face_recognition import get_cos_distance


def calculate_face_embeds(args):
    ## init insightface model
    app = FaceAnalysis(name=f'/home/mingjiahui/.insightface/models/{args.is_model}/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    for ckpt_path in tqdm(args.ckpt_paths):
        target_img_dir = os.path.join(ckpt_path, 'test_sampling')
        result_save_path = os.path.join(target_img_dir, "cos_score.json")
        if not os.path.exists(target_img_dir):
            continue

        result = {}
        for source_img_cache in tqdm(source_img_cache_list):
            source_cache_name = os.path.basename(source_img_cache)
            suffix = source_cache_name.split('.')[-1]
            target_img_path = os.path.join(target_img_dir, source_cache_name.replace(f"--{args.is_model}.{suffix}", '.jpg'))
            if not os.path.exists(target_img_path):
                continue

            source_face_emb = np.load(source_img_cache)
            target_face_emb_path = os.path.join(target_img_dir, source_cache_name.replace(suffix, f'npy'))
            if not os.path.exists(target_face_emb_path):
                face_image = Image.open(target_img_path).convert("RGB")
                face_info_list = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
                if len(face_info_list) == 0:
                    print("no face detect")
                    continue
                face_info = sorted(face_info_list, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
                target_face_emb = torch.from_numpy(face_info.embedding).unsqueeze(0).numpy()
                np.save(target_face_emb_path, target_face_emb)
            else:
                target_face_emb = np.load(target_face_emb_path)
            
            cos_score = get_cos_distance(source_face_emb, target_face_emb)
            result[source_cache_name.split('.')[0]] = cos_score
        
        with open(result_save_path, "w")as f:
            json.dump(result, f)
        print(f"result has saved in {result_save_path}")


def statistical_similarity_change(args):
    os.makedirs(args.plot_save_dir, exist_ok=True)
    result = {}
    source_img_names = []
    source_img_dirs = [
        "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/famous",
        "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/average_id",
    ]
    for source_img_dir in source_img_dirs:
        source_img_names += [name.split('.')[0] for name in os.listdir(source_img_dir) if args.is_model in name]
    for name in source_img_names:
        result.setdefault(name, {})
        for attribute in ['step', 'score']:
            result[name].setdefault(attribute, [])
    
    print("loading cos sim score of training.....")
    for ckpt_path in tqdm(args.ckpt_paths):
        assert 'train_from_step' in ckpt_path or 'scratch' in ckpt_path, ValueError("some error has happened")
        start_step = os.path.dirname(ckpt_path).split('train_from_step')[-1] if 'train_from_step' in ckpt_path else 0
        end_step = os.path.basename(ckpt_path).split('checkpoint-')[-1]
        cos_sim_json_dir = os.path.join(ckpt_path, 'test_sampling')
        if not os.path.exists(cos_sim_json_dir):
            continue
        cos_sim_json = os.path.join(cos_sim_json_dir, 'cos_score.json')
        with open(cos_sim_json, 'r')as f:
            data = json.load(f)
            for k, v in data.items():
                result[k]['step'].append(int(start_step)+int(end_step))  
                result[k]['score'].append(v)  
    
    for test_name, info in result.items():
        steps = info['step']
        scores = info['score']
        plt.figure()
        plt.plot(steps, scores)
        plt.xlabel('step')
        plt.ylabel('score')
        plt.grid(True)
        save_path = os.path.join(args.plot_save_dir, test_name+'.jpg')
        plt.savefig(save_path)
        plt.close
        print(f"result has saved in {save_path}")
    save_path = os.path.join(args.plot_save_dir, 'result.json')
    with open(save_path, 'w')as f:
        json.dump(result, f)
    print(f"result has saved in {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", nargs='+', type=str, help="different version model storage dir", default=None)
    parser.add_argument("--ckpt_paths", type=str, help="checkpoint input dir", default=None)
    parser.add_argument("--is_model", type=str, default="antelopev2")
    parser.add_argument("--mode", type=str, default="calculate_face_embeds",
        help="Union['calculate_face_embeds', 'stat_simi_change']")
    parser.add_argument("--plot_save_dir", type=str, 
        default="/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/face_similarity_result/plot_result")
    args = parser.parse_args()

    def get_ckpt_paths(args):
        model_source_dir = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/finetune/instantid-sdxl-diff_faceid_token"
        args.input_dir = [
            f"{model_source_dir}/20240421-sdxl--V3_4100000--batch_32--lr1e-5--train_from_scratch",
            f"{model_source_dir}/20240422-sdxl--V3_4100000--batch_32--lr1e-5--train_from_step32000",
            f"{model_source_dir}/20240423-sdxl--V3_4100000--batch_32--lr1e-5--train_from_step46000",
        ] if args.input_dir is None else args.input_dir
        args.input_dir = sorted(args.input_dir)

        if args.ckpt_paths is None:
            args.ckpt_paths = []
            for input_dir_ in args.input_dir:
                args.ckpt_paths += [os.path.join(input_dir_, name) for name in os.listdir(input_dir_) \
                    if os.path.isdir(os.path.join(input_dir_, name)) and 'checkpoint' in name]
        return args

    if args.mode=="calculate_face_embeds":
        ## get source image
        source_img_cache_list = []
        source_img_dirs = [
            "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/famous",
            "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/average_id",
        ]
        for source_img_dir in source_img_dirs:
            source_img_cache_list += [os.path.join(source_img_dir, name) for name in os.listdir(source_img_dir) \
                if args.is_model in name]
        args = get_ckpt_paths(args)  
        print(f"ckpt path num:{len(args.ckpt_paths)}")
        calculate_face_embeds(args)

    elif args.mode=="stat_simi_change":
        args = get_ckpt_paths(args)
        statistical_similarity_change(args)
