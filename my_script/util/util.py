from typing import Any, Union
import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

import cv2
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.utils import face_align


class LayerStatus:
    def __init__(self,):
        self.min, self.max, self.mean, self.var = {}, {}, {}, {}
        self.step = None
        self.layer_index = None
    
    def set_step(self, step) -> None:
        self.step = step.item() if isinstance(step, torch.Tensor) else step
        self.min.setdefault(self.step, {})
        self.max.setdefault(self.step, {})
        self.mean.setdefault(self.step, {})
        self.var.setdefault(self.step, {})
    
    # def set_layer_index(self, layer_index) -> None:
    #     self.layer_index = layer_index.item() if isinstance(layer_index, torch.Tensor) else layer_index
    #     self.min[self.step].setdefault(self.layer_index, {})
    #     self.max[self.step].setdefault(self.layer_index, {})
    #     self.mean[self.step].setdefault(self.layer_index, {})
    #     self.var[self.step].setdefault(self.layer_index, {})

    def __call__(self, key, min, max, mean, var) -> None:
        # self.set_layer_index(layer_index)
        self.min[self.step].setdefault(key, [])
        self.max[self.step].setdefault(key, [])
        self.mean[self.step].setdefault(key, [])
        self.var[self.step].setdefault(key, [])

        self.min[self.step][key].append(min.item())
        self.max[self.step][key].append(max.item())
        self.mean[self.step][key].append(mean.item())
        self.var[self.step][key].append(var.item())
    
    def save(self, save_path):
        result = {'max': self.max, 'min': self.min, 'mean': self.mean, 'var': self.var}

        assert os.path.splitext(save_path)[1]=='.json'
        with open(save_path, 'w') as f:
            json.dump(result, f)
        

def visualize_layerstatus(json_path, mode='3d', save_path=r"./data/layer_status0-2d.jpg", ip_scale=1.0):
    with open(json_path, 'r') as f:
        data = json.load(f)

    x = np.arange(70)  # layer index 70
    y = []      # denoising step    20
    statistical_char_result = {}        # { 'max': {'hidden_states': <list>} }
    for char, statistical_char_data in data.items():
        y = []  
        statistical_char_result.setdefault(char, {})
        for step, step_data, in statistical_char_data.items():       # min-dict
            y.append(float(step))
            for obj, obj_data in step_data.items():    # hidden_states-dict
                assert isinstance(obj_data, list)
                statistical_char_result[char].setdefault(obj, [])
                statistical_char_result[char][obj].append(obj_data)

    x_grid, y_grid = np.meshgrid(np.array(x), y)

    assert len(statistical_char_result.keys())==4
    fig = plt.figure(figsize=(10, 8))

    for i, (char, data_group) in enumerate(statistical_char_result.items()):
        z = np.array(data_group['hidden_states'])-np.array(data_group['ip_hidden_states'])
        if mode=='3d':
            ax = fig.add_subplot(int(f"22{i+1}"), projection='3d')
            ax.plot_surface(x_grid, y_grid, z, cmap='viridis')
            ax.set_title(f'{char}')
            ax.set_xlabel('layer index')
            ax.set_ylabel('step')

            # # gen dense
            # interp_func = interp2d(x, y, z, kind='cubic')
            # x_dense = np.linspace(1, 70, 300)
            # y_dense = np.linspace(y.min(), y.max(), 300)
            # x_dense_grid, y_dense_grid = np.meshgrid(x_dense, y_dense)
            # z_dense = interp_func(x_dense, y_dense)

        elif mode=='2d':
            ax = fig.add_subplot(int(f"22{i+1}"))
            ax.plot(x, np.array(data_group['ip_hidden_states']).mean(axis=0)*ip_scale, 'r--', label='image')
            ax.plot(x, np.array(data_group['hidden_states']).mean(axis=0), 'b-', label='text')
            ax.set_title(f'{char}')
            ax.set_xlabel('layer index')
            ax.legend()
        else:
            print('only support 3d and 2d plot')
            exit(0)

    plt.tight_layout()
    plt.savefig(save_path)


class FaceidAcquirer(): 
    def __init__(self) -> None:
        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_face_embeds(self, image: Union[np.ndarray, Image.Image]):
        if isinstance(image, Image.Image):
            image = np.array(image)[:, :, ::-1]
        faces = self.app.get(image)
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        face_image = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224) # you can also segment the face
        return faceid_embeds, face_image
    
    def get_multi_embeds(self, image_paths:list):
        faceid_embeds = []
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        for image_path in image_paths:
            image = cv2.imread(image_path)
            faces = self.app.get(image)
            faceid_embeds.append(torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(0))
        faceid_embeds = torch.cat(faceid_embeds, dim=1)
        return faceid_embeds


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid





    