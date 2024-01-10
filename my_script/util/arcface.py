import cv2
import numpy as np
from typing import Union
from PIL import Image
import os
import shutil
from tqdm import tqdm

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from insightface.utils import face_align


class FacesComp:
    def __init__(self, ):
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_faces(self, img):
        faces = self.app.get(img)
        assert len(faces) == 1, "**Error: The face information of the image is not equal to 1"
        return faces

    # def get_similarity_score(self, ):
    #     return np.dot(self.tar_face[0].normed_embedding, self.gen_faces[0].normed_embedding)

    def change_tar_img(self, tar_img: Union[Image.Image, np.ndarray]):
        self.tar_img = cv2.cvtColor(np.array(tar_img), cv2.COLOR_RGB2BGR) if isinstance(tar_img, Image.Image) else tar_img
        self.tar_faces = self.get_faces(self.tar_img)

    def change_gen_img(self, gen_img: Union[Image.Image, np.ndarray]):
        self.gen_img = cv2.cvtColor(np.array(gen_img), cv2.COLOR_RGB2BGR) if isinstance(gen_img, Image.Image) else gen_img
        self.gen_faces = self.get_faces(self.gen_img)

    def __call__(self, gen_img, tar_img=None):
        if tar_img is not None:
            self.change_tar_img(tar_img)
        self.change_gen_img(gen_img)
        return np.dot(self.tar_faces[0].normed_embedding, self.gen_faces[0].normed_embedding)
    
    def crop_face(self, img: np.ndarray):
        self.change_tar_img(img)
        return face_align.norm_crop(img, landmark=self.tar_faces[0].kps, image_size=224)


if __name__ == '__main__':
    # # 1. Official Code:
    # # https://github.com/deepinsight/insightface/tree/master/python-package
    # app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # app.prepare(ctx_id=0, det_size=(640, 640))
    # img = ins_get_image('t1')
    # cv2.imwrite("./arcface_input.jpg", img)
    # faces = app.get(img)
    # rimg = app.draw_on(img, faces)
    # cv2.imwrite("./t1_output.jpg", rimg)

    # # 2. Test Code ———— get cos similarity score:
    # faces_comp = FacesComp()
    # tar_img = Image.open(r'')
    # gen_img = Image.open(r'')
    # score = faces_comp()
    # print(f"score:{score}")

    # 3. Test Code ———— get crop image(face id):
    faces_comp = FacesComp()
    image_dir = r'/home/mingjiahui/project/sdxl-lora-training/data/face_iplora/image/wangbaoqiang-few_shot-3'
    save_dir = r'/home/mingjiahui/project/sdxl-lora-training/data/face_iplora/image/wangbaoqiang-few_shot-3-crop'
    os.makedirs(save_dir, exist_ok=True)
    flip = True

    file_paths = [os.path.join(image_dir, name) for name in os.listdir(image_dir)]
    for file_path in tqdm(file_paths):
        filename = os.path.basename(file_path)
        save_path = os.path.join(save_dir, filename)
        try:
            if file_path.endswith('.jpg'):
                ori_img = cv2.imread(file_path)
                crop_face = faces_comp.crop_face(ori_img)
                cv2.imwrite(save_path, crop_face)
                if flip:
                    crop_face_flip = cv2.flip(crop_face, 1)
                    flip_save_path = os.path.join(save_dir, 'flip_'+filename)
                    cv2.imwrite(flip_save_path, crop_face_flip)
            elif file_path.endswith('.txt'):
                shutil.copy(file_path, save_path)
            else:
                print("**Error: the file suffix name is not in ['.jpg', '.txt']")
        except Exception as e:
            print(f'{e} ==> {file_path}')
            exit(0)





