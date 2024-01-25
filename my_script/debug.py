
import json
from tqdm import tqdm
import random
data_type = ['Laion', 'coyo', 'ffhq']
statistic_num = [0]*3
# path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1.json"
path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_crop.json"
with open(path, 'r')as f:
    data_list = json.load(f)
    print(len(data_list))
# random.shuffle(data_list)
# data_list = data_list[:10]
for data in tqdm(data_list):
    image_file = data['image_file']
    embeds_path = data['embeds_path']
    crop_reso = data['crop_reso']

    # print(image_file)
    for i, data_type_ in enumerate(data_type):
        if data_type_ in image_file:
            statistic_num[i] += 1
            break
print(statistic_num)

# import cv2
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# transform = transforms.Resize(1024)
# image_path = "/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/data/data-50m/00000/00053/000534559.jpg"
# json_path = "/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/data/data-50m_arcface/00000/00053/000534559.json"
# with open(json_path)as f:
#     data = json.load(f)
#     bbox = data['bbox']
#     lx, ly, rx, ry = bbox
#     image = Image.open(image_path)
#     w, h = image.size
#     resize_ratio = min(w, h)/1024
#     image = transform(image)
#     w, h = image.size

#     l_top_x = max(0, lx)
#     l_top_y = max(0, ly)
#     r_bottom_x = min(w, rx)
#     r_bottom_y = min(h, ry)

#     b_w, b_h = (r_bottom_x - l_top_x), (r_bottom_y - l_top_y)
#     center_x = l_top_x + b_w//2
#     center_y = l_top_y + b_h//2

#     # new_size = min(min(b_w, b_h) * factor, min(w, h))
#     new_size = max(b_w, b_h) * 2
#     x_start = max(0, center_x-new_size//2)
#     y_start = max(0, center_y-new_size//2)
#     x_end = min(w, center_x+new_size//2)
#     y_end = min(h, center_y+new_size//2)
#     if x_start==0 or y_start==0:
#         x_end = min(w, x_start+new_size)
#         y_end = min(h, y_start+new_size)
#     elif x_end==w or y_end==h:
#         x_start = max(0, x_end-new_size)
#         y_start = max(0, y_end-new_size)
#     ratio = (x_end - x_start) / (y_end - y_start)
#     assert ratio==1

#     new_w, new_h = x_end-x_start, y_end-y_start
    
# image = np.array(image)
# cv2.rectangle(image, (int(x_start),int(y_start)), (int(x_start+new_w), int(y_start+new_h)), (0,255,0), 2)
# image = Image.fromarray(image)
# new_w, new_h = image.size
# image.resize((int(new_w*resize_ratio), int(new_h*resize_ratio))).save("./data/other/debug.jpg")
# # image.save("./data/other/debug.jpg")