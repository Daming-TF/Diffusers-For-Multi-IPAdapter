import argparse
import os
import json
from tqdm import tqdm

from PIL import Image
import cv2
import numpy as np
from numpy.linalg import norm

from insightface.app import FaceAnalysis
from torchvision import transforms


def load_LFW():
    result = []
    data_dir = "/mnt/nfs/file_server/public/mingjiahui/data/LFW/lfw-deepfunneled"
    diff_person_dirs = [os.path.join(data_dir, name) for name in os.listdir(data_dir)]
    for diff_person_dir in diff_person_dirs:
        result.append(diff_person_dir) if len(os.listdir(diff_person_dir)) > 1 else None
    print(f"person num:{len(result)}")
    return result


class InsightFaceRec:
    def __init__(self, model_name):
        assert os.path.basename(model_name) in ['antelopev2', 'buffalo_l']
        self.app = FaceAnalysis(name='/home/mingjiahui/.insightface/models/antelopev2/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.transfer = transforms.Resize(768)

    def __call__(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.transfer(img)
        face_info_list = self.app.get(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        if len(face_info_list) == 0:
            return None, None
        face_info = sorted(face_info_list, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        return face_info.embedding, face_info.norm_embedding


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", "--insight_face_model_name", type=str, dest="insight_face_model_name", required=True)
    parser.add_argument("--save_path", type=str, default="/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/debug/cos_distance.json")
    args = parser.parse_args()

    app = InsightFaceRec(args.insight_face_model_name)
    img_dirs = load_LFW()
    result = {}
    total_distance = 0
    for img_dir in tqdm(img_dirs):
        person_name = os.path.basename(img_dir)
        img_paths = [os.path.join(img_dir, name) for name in os.listdir(img_dir)]
        reference_embeds, _ = app(img_paths[0])
        if reference_embeds is None:
            continue
        avg_distance = 0
        for target_path in img_paths[1:]:
            target_embeds, _ = app(target_path)
            if target_embeds is None:
                break
            l2_d = np.dot(reference_embeds, target_embeds)/(norm(reference_embeds)*norm(target_embeds))
            # print(f"l2_d:{l2_d}")
            avg_distance += l2_d
        
        if avg_distance == 0:
            continue
        
        result[person_name] = avg_distance / len(img_paths[1:])
        total_distance += avg_distance / len(img_paths[1:])
    result['result'] = total_distance/len(img_dirs)
    print(f"LFW dataset average distance :{total_distance/len(img_dirs)}")

    with open(args.save_path, 'w')as f:
        json.dump(result, f)
    print(f"result has saved in {args.save_path}")
