import cv2
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import numpy as np
from PIL import Image
import torch
from typing import Union

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))


def get_face_embeds(image: Union[np.ndarray, Image.Image]):
    if isinstance(image, Image.Image):
        image = np.array(image)[:, :, ::-1]
    faces = app.get(image)
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224) # you can also segment the face
    return faceid_embeds, face_image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


if __name__ == '__main__':
    image = cv2.imread("person.jpg")
    get_face_embeds(image)
    