# import pandas as pd

# data = {'Column1': ['Hello', 'Python', 'sd'],
#         'Column2': ['World', 'Pandas', 'dd']}
# df = pd.DataFrame(data)

# df.index = ['11', '22', '33']
# df.columns = ['A', 'B']
# df.to_excel('./data/other/example.xlsx', index=False)


# import xlsxwriter as xw
# workbook = xw.Workbook('./data/other/example.xlsx')
# worksheet1 = workbook.add_worksheet("sheet1")
# worksheet1.activate()

# bold = workbook.add_format({
#     'bold': True,  # 字体加粗
#     'border': 3,  # 单元格边框宽度
#     'align': 'center',  # 水平对齐方式
#     'valign': 'vcenter',  # 垂直对齐方式
#     'fg_color': '#F4B084',  # 单元格背景颜色
#     'text_wrap': True,  # 是否自动换行
# })

# worksheet1.set_column('A:L', 15)
# iou_list = ['0.50:0.95', '0.50', '0.75']

# worksheet1.merge_range('A1:A3', 'Model', bold)
# worksheet1.merge_range('B1:B3', 'Video_name', bold)
# worksheet1.merge_range('C1:G1', 'AP', bold)
# worksheet1.merge_range('H1:L1', 'AR', bold)

# worksheet1.merge_range('C2:E2', 'Area=all', bold)
# worksheet1.write('F2', 'Area=medium', bold)
# worksheet1.write('G2', 'Area=large', bold)

# worksheet1.merge_range('H2:J2', 'Area=all', bold)
# worksheet1.write('K2', 'Area=medium', bold)
# worksheet1.write('L2', 'Area=large', bold)

# worksheet1.write_row('C3', iou_list, bold)
# worksheet1.write('F3', '0.50:0.95', bold)
# worksheet1.write('G3', '0.50:0.95', bold)
# worksheet1.write_row('H3', iou_list, bold)
# worksheet1.write('K3', '0.50:0.95', bold)
# worksheet1.write('L3', '0.50:0.95', bold)

# worksheet1.merge_range('A4:A14', 'baidu', bold)

# workbook.close()


# from openpyxl import load_workbook
# wb = load_workbook('/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/finetune/20140130-sd15-crop--V1-wo_xformer-scratch/checkpoint-2000/result.xlsx')
# sheet = wb.active
# cell_value = sheet['B67'].value
# print(type(cell_value))
# print("B67单元格的值是:", cell_value)

# import torch
# tensors = [torch.randn(1, 3, 224, 224) for _ in range(32)]
# a = torch.stack(tensors, dim=1)
# print(a.shape)
# b = torch.cat(tensors, dim=0)
# print(b.shape)

# import torch
# pt_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/finetune/portrait-ti_token/20240208-sd15-crop--V1-wo_xformer-scratch/checkpoint-32500/sd15_faceid_portrait.bin"
# sd = torch.load(pt_path)
# print(sd.keys())
# a = sd['text_proj']
# print(a.keys())

# from datasets import load_dataset

# folder = 'all_test_data'
# folder_dir = "./data/"

# dataset = load_dataset(
#     folder,
#     data_dir=folder_dir,
#     cache_dir='./'
# )

# print(dataset.keys())

# import os
# import sys
# current_path = os.path.dirname(__file__)
# sys.path.append(os.path.dirname(current_path))
# from data.get_all_face_info import json_transfer
# json_transfer()

# import json
# import os
# json_dir = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/_tmp/error_image/"
# json_paths = [os.path.join(json_dir, name) for name in os.listdir(json_dir)]
# for i, json_path in enumerate(json_paths):
#     with open(json_path, 'r')as f:
#         data = json.load(f)
#         print(f"{i}:{len(data)}")

# import torch
# # from safetensors.torch import load_file
# # model_path = "/mnt/nfs/file_server/public/mingjiahui/models/lllyasviel--control_v11p_sd15_openpose/diffusion_pytorch_model.fp16.bin"
# # sd0 = torch.load(model_path)

# # ours_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/finetune/instantid/20240211-sd15--V1-wo_xformer-scratch/checkpoint-1/diffusion_pytorch_model.safetensors"
# # sd1 = load_file(ours_path, device="cpu")
# # print(set(sd0) == set(sd1))

# tensor = torch.randn(3, 4, 5)
# print(tensor.size())
# print(tensor.size(0))
# print(tensor.shape)

# from safetensors.torch import load_file
# import torch
# dir = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/finetune/instantid/20240213-sd15--V1-wo_xformer-pretrain_from_step_13500/"
# path0 = f"{dir}/checkpoint-1/diffusion_pytorch_model.safetensors"
# path1 = f"{dir}/checkpoint-12000/diffusion_pytorch_model.safetensors"
# sd0 = load_file(path0)
# sd1 = load_file(path1)
# print(set(sd0.keys()) == set(sd1.keys()))
# for key in sd0.keys():
#     if not torch.allclose(sd0[key], sd1[key]):
#         print(f"difference\t{sd0[key].mean()}\t{sd1[key].mean()}")
#         break
#     # if sd0[key] != sd1[key]:
#     #     print("difference")

# import os
# import sys
# current_dir = os.path.dirname(__file__)
# sys.path.append(os.path.dirname(current_dir))
# from my_script.deepface.eval_model import inference_instantid
# save_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/finetune/instantid/20240213-sd15--V1-wo_xformer-pretrain_from_step23000/checkpoint-132000/"
# inference_instantid(save_path, 'sd15_instantid.bin')

# # 测试图片反标准化
# from torchvision import transforms
# from PIL import Image
# import torchvision.transforms.functional as TF
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5]),
# ])
# image_path="/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data/aoteman.jpg"
# image = Image.open(image_path)
# image_tensor = transform(image)

# image_tensor = image_tensor * 0.5 + 0.5
# pil_image = TF.to_pil_image(image_tensor)
# save_path = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/other/debug.jpg"
# pil_image.save(save_path)
# print(f"result has saved in {save_path}")

# # 测试图片叠加测试
# from PIL import Image
# import numpy as np
# import cv2
# image_path0 = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data/aoteman.jpg"
# image_path1 = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data/luxun.jpg"
# # image0 = Image.open(image_path0).convert("RGB").resize((512,512))
# # image1 = Image.open(image_path1).convert("RGB").resize((512,512))
# image0 = cv2.resize(cv2.imread(image_path0), (512, 512))
# image1 = cv2.resize(cv2.imread(image_path1), (512, 512))
# print(image0.shape)
# print(image1.shape)

# # alpha = 1/3
# # image1 = cv2.addWeighted(image1, alpha, np.zeros(image1.shape, image1.dtype), 0, 0)
# result = cv2.addWeighted(image0, 0.8, image1, 0.4, 0)
# save_path = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/other/debug.jpg"
# cv2.imwrite(save_path, result)
# print(f"result has saved in {save_path}")


# import json
# import cv2
# from PIL import Image
# import numpy as np
# # json_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_with_all_face_info.json"
# # with open(json_path, 'r')as f:
# #     data = json.load(f)
# # metadata = data[0]
# # print(metadata.keys())
# # print(metadata['face_info_json'])

# # 1.init
# json_path1="/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/data/data-50m-V1_face_all_info/00000/00038/000382716.json"
# with open(json_path1)as f:
#     data = json.load(f)

# image_path = data['image_path']
# bbox = data['bbox'][0]
# x1, y1, x2, y2 = bbox
# kps = np.array(data['kps'][0])
# print(f"kps shape:{np.array(kps).shape}")

# image = cv2.imread(image_path)
# h, w , _ = image.shape
# image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
# save_path = "./data/other/debug.jpg"

# # 2.expand according to the bbox area
# factor = 2
# x1, y1, x2, y2 = bbox
# x1 = max(0, x1)
# y1 = max(0, y1)
# x2 = min(w, x2)
# y2 = min(h, y2)
# bb_w, bb_h = x2-x1, y2-y1
# cx = x1 + bb_w // 2
# cy = y1 + bb_h // 2
# # adaptive adjustment
# crop_size = max(bb_w, bb_h)*factor
# x1 = max(0, cx-crop_size//2)
# y1 = max(0, cy-crop_size//2)
# x2 = min(w, cx+crop_size//2)
# y2 = min(h, cy+crop_size//2)
# if x2==w:
#     x1 = max(0, x2-crop_size)
# if y2==h:
#     y1 = max(0, y2-crop_size)
# if x1==0:
#     x2 = min(w, x1+crop_size)
# if y1==0:
#     y2 = min(h, y2+crop_size)
# # cut square area
# w, h = x2-x1, y2-y1
# image = image[int(y1):int(y2), int(x1):int(x2)]
# # fix kps
# kps[:, 0] = kps[:, 0] - x1
# kps[:, 1] = kps[:, 1] - y1


# # 3.short side resize
# size = 512
# if h < w:
#     new_h = size
#     new_w = int(new_h * (w / h))
# else:
#     new_w = size
#     new_h = int(new_w * (h / w))
# image = cv2.resize(image, (new_w, new_h))
# # top = int(((new_h - size) // 2) * 1/5)
# top = 0
# left = (new_w - size) // 2
# image = image[top:top+size, left:left+size]
# kps[:, 0] = (kps[:, 0] * new_w / w) - left
# kps[:, 1] = (kps[:, 1] * new_h / h) - top

# # 4.draw points
# for point in kps:
#     cv2.circle(image, (int(point[0]), int(point[1])), 5, (0,255,0), -1)
# cv2.imwrite(save_path, image)
# print(f"result has saved in {save_path}")


# import os
# import sys
# current_dir = os.path.dirname(__file__)
# sys.path.append(os.path.dirname(current_dir))
# from my_script.deepface.eval_model import inference_instantid

# checkout_dir = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/finetune/instantid-portrait/20240222-sd15--V1-wo_xformer-pretrain_from_step120000/checkpoint-1/"
# inference_instantid(checkout_dir, 'sd15_instantid.bin')


# from PIL import Image

# image_path = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data/guonan.jpg"
# image = Image.open(image_path)
# result = image.convert("L")
# result.save("/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/other/debug.jpg")


# 得到不同尺度的大头像
# from PIL import Image
# import numpy as np
# from insightface.app import FaceAnalysis
# from torchvision import transforms
# import cv2
# import torch

# import os
# import sys
# current_dir = os.path.dirname(__file__)
# sys.path.append(os.path.dirname(current_dir))
# from my_script.deepface.eval_model import resize_and_crop

# transform = transforms.Resize(512)

# image_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/backups/test_data_V2-fix_prompt/black/black_1_0.webp"
# save_dir = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/other/crop_diff_factor"
# os.makedirs(save_dir, exist_ok=True)
# image_id = os.path.basename(image_path).split('.')[0]
# app = FaceAnalysis(name='/home/mingjiahui/.insightface/models/buffalo_l/', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))

# face_image = Image.open(image_path).convert("RGB")
# face_image = transform(face_image)
# face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
# face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]
# face_emb = torch.from_numpy(face_info.normed_embedding).unsqueeze(0).unsqueeze(0)
# []
# factor_list = np.arange(2, 10+2, 2).tolist()
# for factor in factor_list:
#     cropped_image, kps = resize_and_crop(face_image, face_info['kps'], face_info['bbox'].tolist(), factor=factor, size=1024)
#     save_path = os.path.join(save_dir, image_id+f'_crop{factor}.jpg')
#     cv2.imwrite(save_path, cropped_image)
#     print(f"result has saved in {save_path}")


# import json
# import numpy as np
# json_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_crop.json"
# with open(json_path, 'r')as f:
#     data = json.load(f)
# print(data[0].keys())
# print(data[0]['image_file'])
# print(data[0]['crop_reso'])
# print(data[0]['embeds_path'])
# embeds = np.load(data[0]['embeds_path'])
# print(embeds.shape)



# import pandas as pd
# import numpy as np
# df = pd.read_csv("/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/other/excel/style_online.csv", encoding='gbk')
# print(type(df))
# print(df.keys())

# lora_name = df['风格名称']
# before_prompt = df['before_prompt']

# print(type(before_prompt))
# print(len(before_prompt))
# print(before_prompt.values[:3])

# for prompt in before_prompt.values[:3]:
#     if prompt != float('nan'):
#         print(prompt, type(prompt))
        

# import torch
# from torchvision import transforms
# a = torch.randint(low=0, high=255, size=(3,512,512)).float()
# # print(a)
# print(a.mean())
# print(a.std())
# transform = transforms.Normalize([0.5], [0.5])
# b = transform(a)
# print(b.mean())
# print(b.std())
# print(b.shape)
# print(b[0].mean())
# print(b.std())


import torch
import torch.nn as nn
logvar_init = 0.0
logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
print(logvar)
rec_loss = 1
nll_loss = rec_loss / torch.exp(logvar) + logvar
print(rec_loss)