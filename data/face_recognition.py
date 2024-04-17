import argparse
import os
import json
from tqdm import tqdm
import sys
import torch

from PIL import Image
import cv2
import numpy as np
from numpy.linalg import norm

from torchvision import transforms


def load_LFW():
    result = []
    data_dir = "/mnt/nfs/file_server/public/mingjiahui/data/LFW/lfw-deepfunneled"
    diff_person_dirs = [os.path.join(data_dir, name) for name in os.listdir(data_dir)]
    for diff_person_dir in diff_person_dirs:
        result.append(diff_person_dir) if len(os.listdir(diff_person_dir)) > 1 else None
    print(f"person num:{len(result)}")
    return result


def load_self(input):
    img_paths = [os.path.join(input, name) for name in os.listdir(input) if name.endswith('.jpg')]
    img_paths.sort()
    return img_paths


def transfer_embeds(embeds):
    if embeds is None:
        return None
    if not torch.is_tensor(embeds):
        embeds = torch.tensor(embeds)
    embeds = embeds.unsqueeze(0) if len(embeds.shape) == 1 else embeds
    assert len(embeds.shape) == 2, ValueError("some error has happened")
    return embeds.cpu().numpy()


def get_l2_distance(embed_0, embed_1):
    l2_distance = np.linalg.norm(embed_0 - embed_1, ord=2, axis=1)
    return l2_distance.mean()


def get_cos_distance(embed_0, embed_1):
    dot_product = np.sum(embed_0 * embed_1, axis=1)
    embed_0_norm = np.linalg.norm(embed_0, axis=1)
    embed_1_norm = np.linalg.norm(embed_1, axis=1)
    cosine_sim = dot_product / (embed_0_norm * embed_1_norm)
    distance = np.array([1 - l for l in cosine_sim]).mean()
    return distance


class InsightFaceRec:
    def __init__(self, model_name):
        from insightface.app import FaceAnalysis
        assert os.path.basename(model_name) in ['antelopev2', 'buffalo_l']
        self.app = FaceAnalysis(name='/home/mingjiahui/.insightface/models/antelopev2/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.transfer = transforms.Resize(512)

    def __call__(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.transfer(img)
        face_info_list = self.app.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        if len(face_info_list) == 0:
            print(f"no face detect ==> {img_path}")
            return None, None
        face_info = sorted(face_info_list, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        return transfer_embeds(face_info.embedding), transfer_embeds(face_info.normed_embedding)
    

class Face2DiffRec:
    def __init__(self,):
        import face_alignment
        sys.path.append("/home/mingjiahui/project")
        from Face2Diffusion.inference_f2d import msid_base_patch8_112
        from Face2Diffusion.src.utils import align, warp_img
        from Face2Diffusion.src.modules import IMG2TEXTwithEXP
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # init detector
        self.detector=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cuda' if torch.cuda.is_available() else 'cpu')
        # init text2img exp
        w_map = "/mnt/nfs/file_server/public/mingjiahui/models/Face2Diffusion/mapping.pt"
        self.img2text = IMG2TEXTwithEXP(384*4,384*4,768)
        self.img2text.load_state_dict(torch.load(w_map,map_location='cpu'))
        self.img2text = self.img2text.to(self.device)
        self.img2text.eval()
        # init msid
        w_msid = "/mnt/nfs/file_server/public/mingjiahui/models/Face2Diffusion/msid.pt"
        self.msid = msid_base_patch8_112(ext_depthes=[2,5,8,11])
        self.msid.load_state_dict(torch.load(w_msid))
        self.msid=self.msid.to(self.device)
        self.msid.eval()
        self.align = align
        self.warp_img = warp_img

    def __call__(self, img_path, return_mlp_output=True):
        lmk = self.detector.get_landmarks(img_path)
        if lmk is None:
            print(f"no face detect ==> {img_path}")
            return None, None
        lmk=np.array(lmk)[0]
        img = np.array(Image.open(img_path).convert('RGB'))
        with torch.no_grad():
            M=self.align(lmk)
            img=self.warp_img(img,M,(112,112))/255
            img=torch.tensor(img).permute(2,0,1).unsqueeze(0)
            img=(img-0.5)/0.5
            idvec = self.msid.extract_mlfeat(img.to(self.device).float(),[2,5,8,11])
            if return_mlp_output:
                tokenized_identity_first, tokenized_identity_last = self.img2text(idvec,exp=None)
                mlp_output_tensor = torch.cat([tokenized_identity_first, tokenized_identity_last], dim=0)
                mlp_output = transfer_embeds(mlp_output_tensor)
                return mlp_output, mlp_output/np.linalg.norm(mlp_output, ord=2, axis=1, keepdims=True)
        return transfer_embeds(idvec), transfer_embeds(idvec)/np.linalg.norm(transfer_embeds(idvec), ord=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", "--insight_face_model_name", type=str, dest="insight_face_model_name", 
                        default='/home/mingjiahui/.insightface/models/antelopev2')
    parser.add_argument("--save_dir", type=str, default="/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/expression/debug")
    parser.add_argument("--app_mode", type=str, default='if', help="Union[ 'if', 'f2d' ]")
    parser.add_argument("--dis_mode", type=str, default='cos', help="Union[ 'cos', 'l2' ]")
    parser.add_argument("--d", type=str, default='lfw', help="Union[ 'lfw', 'mjh', 'xy ]")
    args = parser.parse_args()

    if args.app_mode == 'if':
        app = InsightFaceRec(args.insight_face_model_name)
    elif args.app_mode == 'f2d':
        app = Face2DiffRec()
    else:
        ValueError("param 'app_mode' is invalid")

    key = f'{args.app_mode}_{os.path.basename(args.insight_face_model_name)}' if args.app_mode == 'if' \
            else args.app_mode
    if args.d=='lfw':
        save_key = ['lfw-deepfunneled', f'lfw-deepfunneled-embed--{key}', f'lfw-deepfunneled-norm_embed--{key}']
        # save_key = ['lfw-deepfunneled', f'lfw-deepfunneled-embed-before_mlp--{key}', f'lfw-deepfunneled-norm_embed-before_mlp--{key}']
        img_dirs = load_LFW()  
        img_paths = []
        for img_dir in img_dirs:
            img_paths += [os.path.join(img_dir, name) for name in os.listdir(img_dir) if name.endswith('.jpg')]
    elif args.d=='xy':
        save_key = ['xy_exp', f'xy_exp-embed--{key}', f'xy_exp-norm_embed--{key}']
        # save_key = ['xy_exp', f'xy_exp-embed-before_mlp--{key}', f'xy_exp-norm_embed-before_mlp--{key}']
        input = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/expression/xy_exp"
        img_paths = load_self(input)
    elif args.d=='mjh':
        save_key = ['mjh_exp', f'mjh_exp-embed--{key}', f'mjh_exp-norm_embed--{key}']
        # save_key = ['mjh_exp', f'mjh_exp-embed-before_mlp--{key}', f'mjh_exp-norm_embed-before_mlp--{key}']
        input = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/expression/mjh_exp"
        img_paths= load_self(input)
    else:
        ValueError("param 'd' is invalid")

    def save_embeds(embed, norm_embed, img_path, save_key):
        suffix = os.path.basename(img_path).split('.')[-1]
        save_path_0 = img_path.replace(save_key[0], save_key[1]).replace(f'.{suffix}', '.npy')
        os.makedirs(os.path.dirname(save_path_0), exist_ok=True)
        np.save(save_path_0, embed)
        save_path_1 = img_path.replace(save_key[0], save_key[2]).replace(f'.{suffix}', '.npy')
        os.makedirs(os.path.dirname(save_path_1), exist_ok=True)
        np.save(save_path_1, norm_embed)


    # #### Calculate the LFW score
    # os.makedirs(args.save_dir, exist_ok=True)
    # save_path = os.path.join(args.save_dir, f"{args.d}-{args.app_mode}-{args.dis_mode}--{save_key[2]}.json")
    # result = {}
    # total_distance = 0
    # total_count = 0
    # for img_dir in tqdm(img_dirs):
    #     # get reference embeds
    #     img_paths = [os.path.join(img_dir, name) for name in os.listdir(img_dir)]
    #     img_paths.sort()

    #     ## M1
    #     # reference_embeds, _ = app(img_paths[0])     # {1, n}
    #     # reference_embeds = transfer_embeds(reference_embeds)
    #     # if reference_embeds is None:
    #     #     continue
    #     # save_embeds(reference_embeds, img_paths[0], save_key)
    #     ## M2
    #     key_index = 2 if 'norm' in os.path.basename(save_path) else 1
    #     suffix = os.path.basename(img_paths[0]).split('.')[-1]
    #     refer_embed_file = img_paths[0].replace(save_key[0], save_key[key_index]).replace(f'.{suffix}', '.npy')
    #     if not os.path.exists(refer_embed_file):
    #         continue
    #     refer_embeds = np.load(refer_embed_file)

    #     # get target embeds
    #     avg_distance = 0
    #     count = 0
    #     for target_path in img_paths[1:]:
    #         ## M1
    #         # target_embeds, _ = app(target_path)     # {1, n}
    #         # target_embeds = transfer_embeds(target_embeds)
    #         # if target_embeds is None:
    #         #     break
    #         # save_embeds(target_embeds, target_path, save_key)
    #         ## M2
    #         suffix = os.path.basename(target_path).split('.')[-1]
    #         target_embed_file = target_path.replace(save_key[0], save_key[key_index]).replace(f'.{suffix}', '.npy')
    #         if not os.path.exists(target_embed_file):
    #             continue
    #         target_embeds = np.load(target_embed_file)
            
    #         if args.dis_mode=='l2':
    #             distance = get_l2_distance(refer_embeds, target_embeds)
    #         elif args.dis_mode=='cos':
    #             distance = get_cos_distance(refer_embeds, target_embeds)
    #         else:
    #             ValueError("Param 'dis_mode' is invalid")

    #         # print(f"l2_d:{l2_d}")
    #         avg_distance += distance
    #         count += 1
        
    #     if avg_distance == 0:
    #         continue
        
    #     person_name = os.path.basename(img_dir)
    #     result[person_name] = avg_distance / count
    #     total_distance += avg_distance / count
    #     total_count += 1

    # result['result'] = total_distance/total_count
    # print(f"LFW dataset average distance :{total_distance/total_count}")

    # with open(save_path, 'w')as f:
    #     json.dump(result, f)
    # print(f"result has saved in {save_path}")
    # # +++++++++++++++++++++++++++++++++++++++++++++++


    # #### get face embeds
    # kwargs = {'return_mlp_output': False} if args.app_mode == 'f2d' else {}
    # for img_path in tqdm(img_paths):
    #     embeds, norm_embeds = app(img_path, **kwargs)
    #     if embeds is None:
    #         continue
    #     save_embeds(embeds, norm_embeds, img_path, save_key)
    # # ++++++++++++++++++++++++++++++++++++++++


    ## Select the reference chart to compare the cosine distance between the two embeds
    # mjh_exp-embed--f2d
    # mjh_exp-embed--if_antelopev2
    # mjh_exp-norm_embed--f2d
    # mjh_exp-norm_embed--if_antelopev2
    # xy_exp-embed--f2d
    # xy_exp-embed--if_antelopev2

    # refer_embed_file = f"/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/expression/mjh_exp-embed--{key}/0_0.npy"
    refer_embed_file = f"/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/expression/mjh_exp-norm_embed--{key}/0_0.npy"
    # refer_embed_file = f"/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/expression/mjh_exp-norm_embed-before_mlp--{key}/0_0.npy"
    print(refer_embed_file)
    refer_embed = np.load(refer_embed_file)

    # save_name = f"comp_cos_dis--{args.d}--mjh_exp-embed--{key}--0_0.json"
    save_name = f"comp_cos_dis--{args.d}--mjh_exp-norm_embed--{key}--0_0.json"
    # save_name = f"comp_cos_dis--{args.d}--mjh_exp-norm_embed-before_mlp--{key}--0_0.json"
    save_path = os.path.join(args.save_dir, save_name)
    key_index = 2 if 'norm' in save_name else 1

    result = {}
    total = 0
    distance = 0
    for img_path in tqdm(img_paths):
        name = os.path.basename(img_path)
        suffix = os.path.basename(img_path).split('.')[-1]
        target_embed_file = img_path.replace(save_key[0], save_key[key_index]).replace(f'.{suffix}', '.npy')
        if not os.path.exists(target_embed_file):
            continue
        
        target_embed = np.load(target_embed_file)
        assert abs(np.linalg.norm(target_embed, ord=2)-1)<0.1, ValueError(f"norm embeds is not invalid ==> {np.linalg.norm(target_embed, ord=2)}")
        cos_distance = get_cos_distance(refer_embed, target_embed)
        result[name] = cos_distance
        if not target_embed_file == refer_embed_file :
            distance += cos_distance
            total += 1
        else:
            print(f"refer embed file:\t{refer_embed_file}")
            
    avg_distance = distance / total
    result['result'] = avg_distance
    print(f"refer npy:{refer_embed_file}\ndata:{args.d}\navg distance:{avg_distance}")
    with open(save_path, 'w')as f:
        json.dump(result, f)
    print(f"result has saved in {save_path}")
    # ++++++++++++++++++++++++++++