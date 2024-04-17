# prompt_template= "a young woman with a {}, wearing a pink shirt. She is standing in front of a fence, possibly in a park or an outdoor setting. The woman appears to be enjoying her time outdoors, possibly engaging in a sport or a recreational activity. "
# expression_leys = ['bright smile', 'sad face', 'astonished face', 'exaggerated expression']
# for expression_ley in expression_leys:
#     print(prompt_template.format(expression_ley))
#     print('\n')

# print(all([1, 1]))

# import torch

# ckpt = "/mnt/nfs/file_server/public/mingjiahui/models/InstantX--InstantID/ip-adapter.bin" 
# sd = torch.load(ckpt) 
# print(sd.keys())


# # choise reso
# import multiprocessing
# import json
# from PIL import Image
# from tqdm import tqdm
# import os
# import random


# def processing(data, i, save_dir):
#     result = []
#     save_path = os.path.join(save_dir, f"{i}.json")
#     for data_ in tqdm(data):
#         image_path = data_['image_file']
#         image = Image.open(image_path)
#         if min(image.size) < 768:
#             continue
#         result.append(data_)
#         # break
#     with open(save_path, 'w')as f:
#         json.dump(result, f)
#     print(f"result has saved in {save_path}")


# if __name__ == '__main__':
#     json_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_with_all_face_info.json"
#     save_dir = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/_tmp/min_reso_512"
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_with_all_face_info--min_reso_768.json"
#     result = []
#     process_num = 4
#     processors = []
    
#     with open(json_path, 'r')as f:
#         data = json.load(f)
#         random.shuffle(data)
        
#     print(f"total data num:{len(data)}")
#     data_index = 0
#     chunk_num = len(data) // process_num
#     residue_num = len(data) % process_num
#     for i in range(process_num):
#         if i < residue_num:
#             chunk_data = data[data_index:data_index+chunk_num+1]
#             data_index = data_index+chunk_num+1
#         else:
#             chunk_data = data[data_index:data_index+chunk_num]
#             data_index = data_index+chunk_num
#         processor = multiprocessing.Process(target=processing, args=(chunk_data, i, save_dir))
#         processors.append(processor)
#         processor.start()
#     for processor in processors:
#         processor.join()

#     for i in range(process_num):
#         json_path = os.path.join(save_dir, f"{i}.json")
#         with open(json_path, 'r')as f:
#             result_ = json.load(f)
#             result += result_
#     with open(save_path, 'w')as f:
#         json.dump(result, f)
#     print(f"result has save in {save_path}\t Total num:{len(result)}")


# json_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V2_mjh_ori.json"
# with open(json_path, 'r')as f:
#     data = json.load(f)
# print(f"total num:{len(data)}") 
# print(type(data[0]))
# print(data[-1].keys())
# print(data[-1]['type'])
# print(data[-1]['source'])
# print(data[-1]['prompt'])


# json_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_with_all_face_info--min_reso_768.json"
# with open(json_path, 'r')as f:
#     data = json.load(f)
# print(data[-1].keys())
# # print(data[-1]['type'])
# # print(data[-1]['source'])
# # print(data[-1]['prompt'])


# json_path = [
#     # "/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/data/_tmp/train_instantid_controlnet-laion_all_one_face.json",
#     # "/mnt/nfs/file_server/public/mingjiahui/data/coyo700m/_tmp/train_instantid_controlnet-coyo_all_one_face.json",
#     # "/mnt/nfs/file_server/public/mingjiahui/data/ffhq/data/decompression_data/_tmp/train_instantid_controlnet-ffhq_all_one_face.json"

#     "/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/data/_tmp/train_V1-laion_all_one_face.json",
#     "/mnt/nfs/file_server/public/mingjiahui/data/coyo700m/_tmp/train_V1-coyo_all_one_face.json",
#     "/mnt/nfs/file_server/public/mingjiahui/data/ffhq/data/decompression_data/_tmp/train_V1-ffhq_all_one_face.json",
# ]
# count = 0
# for json_path_ in json_path:
#     with open(json_path_, 'r')as f:
#         data = json.load(f)
#         print(f"data num:{len(data)}")
#         count += len(data)
# print(count)


# from fastparquet import ParquetFile
# filename = "/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/laion_face_meta/laion_face_part_00000.parquet"
# pf = ParquetFile(filename)

# dF = pf.to_pandas()
# print(f"total num:{len(dF)}")
# for index, row in dF.iterrows():
#     print(row.keys())
#     print(row['URL'])
#     print(row['SAMPLE_ID'])
#     print(row['LICENSE'])
#     print(row['NSFW'])
#     print(row['similarity'])
#     exit(0)


# import json
# json_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_with_all_face_info--antelopev2_non_norm--min_reso_768.json"
# with open(json_path, 'r')as f:
#     data = json.load(f)
#     print(data[0]['embeds_path'])


# import json
# from tqdm import tqdm
# data_dict = {
#         'Laion': ['data-50m_arcface', 'data-50m_antelopev2_non_norm'],
#         'coyo': ['coyo700m/data_arcface', 'coyo700m/data_antelopev2_non_norm'],
#         'ffhq': ['in-the-wild-images_arcface', 'in-the-wild-images_antelopev2_non_norm'],
# }
# result = []
# json_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_with_all_face_info--min_reso_768.json"
# tmp_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/_tmp/antelopev2_non_norm"
# save_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_with_all_face_info--antelopev2_non_norm--min_reso_768.json"
# os.makedirs(tmp_path, exist_ok=True)
# with open(json_path, 'r')as f:
#     data = json.load(f)
#     print(f"total num:{len(data)}")

# def processing(data, i):
#     tmp_save_path = os.path.join(tmp_path, f"{i}.json")
#     for data_ in tqdm(data):
#         find_status = False
#         for k, v in data_dict.items():
#             if k in data_['image_file']:
#                 find_status = True
#                 break
#         assert find_status, ValueError("some error happened")
#         new_embeds_path = data_['embeds_path'].replace(v[0], v[1])
#         if not os.path.exists(new_embeds_path):
#             continue
#         data_['embeds_path'] = new_embeds_path
#         result.append(data_)

#     with open(tmp_save_path, 'w')as f:
#         json.dump(result, f)
#         # print(f"result has saved in {save_path}")
#         # print(f"Total num:{len(result)}")


# data_index = 0
# process_num = 4
# processors = []
# chunk_num = len(data) // process_num
# residue_num = len(data) % process_num
# for i in range(process_num):
#     if i < residue_num:
#         chunk_data = data[data_index:data_index+chunk_num+1]
#         data_index = data_index+chunk_num+1
#     else:
#         chunk_data = data[data_index:data_index+chunk_num]
#         data_index = data_index+chunk_num
#     processor = multiprocessing.Process(target=processing, args=(chunk_data, i))
#     processors.append(processor)
#     processor.start()
# for processor in processors:
#     processor.join()

# for i in range(process_num):
#     json_path = os.path.join(tmp_path, f"{i}.json")
#     with open(json_path, 'r')as f:
#         result_ = json.load(f)
#         result += result_
# with open(save_path, 'w')as f:
#     json.dump(result, f)
# print(f"result has save in {save_path}\t Total num:{len(result)}")


#### for instantid  kps transfer
# import numpy as np
# from PIL import Image
# import cv2
# import math
# size = 512

# def resize_and_crop(image:Image.Image, kps:np.ndarray, bbox=None):
#         # 1.init
#         if isinstance(image, Image.Image):
#             w, h = image.size
#             image = np.array(image)     # [::-1]
#             # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # 2. expand according to the bbox area
#         if bbox is not None:
#             factor = random.randint(2, 6)
#             x1, y1, x2, y2 = bbox
#             x1 = max(0, x1)
#             y1 = max(0, y1)
#             x2 = min(w, x2)
#             y2 = min(h, y2)
#             bb_w, bb_h = x2-x1, y2-y1
#             cx = x1 + bb_w // 2
#             cy = y1 + bb_h // 2
#             # adaptive adjustment
#             crop_size = max(bb_w, bb_h)*factor
#             x1 = max(0, cx-crop_size//2)
#             y1 = max(0, cy-crop_size//2)
#             x2 = min(w, cx+crop_size//2)
#             y2 = min(h, cy+crop_size//2)
#             if x2==w:
#                 x1 = max(0, x2-crop_size)
#             if y2==h:
#                 y1 = max(0, y2-crop_size)
#             if x1==0:
#                 x2 = min(w, x1+crop_size)
#             if y1==0:
#                 y2 = min(h, y2+crop_size)
#             # cut square area
#             w, h = x2-x1, y2-y1
#             image = image[int(y1):int(y2), int(x1):int(x2)]
#             # fix kps
#             kps[:, 0] = kps[:, 0] - x1
#             kps[:, 1] = kps[:, 1] - y1
        
#         # 3.short side resize
#         if h < w:
#             new_h = size
#             new_w = int(new_h * (w / h))
#         else:
#             new_w = size
#             new_h = int(new_w * (h / w))
#         resized_img = cv2.resize(image, (new_w, new_h))
#         # top = (new_h - self.size) // 2
#         top = 0
#         left = (new_w - size) // 2

#         cropped_img = Image.fromarray(resized_img[top:top+size, left:left+size])
#         kps[:, 0] = (kps[:, 0] * new_w / w) - left
#         kps[:, 1] = (kps[:, 1] * new_h / h) - top

#         return cropped_img, (new_w, new_h), kps

# def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
#     kps_list = [kps] if not isinstance(kps, list) else kps
#     stickwidth = 4
#     limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])

#     w, h = image_pil.size
#     out_img = np.zeros([h, w, 3])

#     for kps in kps_list:
#         kps = np.array(kps)
#         for i in range(len(limbSeq)):
#             index = limbSeq[i]
#             color = color_list[index[0]]

#             x = kps[index][:, 0]
#             y = kps[index][:, 1]
#             length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
#             angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
#             polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
#             out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
#         out_img = (out_img * 0.6).astype(np.uint8)

#         for idx_kp, kp in enumerate(kps):
#             color = color_list[idx_kp]
#             x, y = kp
#             out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

#     out_img_pil = Image.fromarray(out_img.astype(np.uint8))
#     return out_img_pil

# image_path = "/mnt/nfs/file_server/public/mingjiahui/data/coyo700m/data/01267/012677562.jpg"
# json_path = "/mnt/nfs/file_server/public/mingjiahui/data/coyo700m/data_V1_face_all_info/01267/012677562.json"
# with open(json_path, 'r')as f:
#     face_info = json.load(f)
# image = Image.open(image_path)
# kps = face_info['kps'][0]
# bbox = face_info['bbox'][0]
# image_kps = draw_kps(image, np.array(kps))
# save_path = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/debug/debug0.jpg"
# image_kps.save(save_path)
# print(f"result has saved in {save_path}")
# cropped_img, (new_w, new_h), kps = resize_and_crop(image, np.array(kps), bbox)
# result = draw_kps(cropped_img, kps)
# save_path = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/debug/debug.jpg"
# result.save(save_path)
# print(f"result has saved in {save_path}")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# import json
# # json_path = "/mnt/nfs/file_server/public/mingjiahui/data/MJ_xy/_tmp/traindata_V2_mjh_ori.json"
# json_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V2_mj_xy.json"
# with open(json_path, 'r')as f:
#     data = json.load(f)
# for data_ in data:
#     print(data_)
#     # print(data_['prompt'])
#     # print(data_['source'])
#     break

# import time
# from tqdm import tqdm
# # i = (0,)*100000000
# i = [0]*100000000
# a = time.time()
# for i_ in tqdm(i):
#     continue
# b = time.time()
# print(b-a)


# import torch.nn as nn
# import torch
# # NLP Example
# batch, sentence_length, embedding_dim = 20, 5, 10
# embedding = torch.randn(batch, sentence_length, embedding_dim)
# layer_norm = nn.LayerNorm(embedding_dim)
# layer_norm.train()
# # Activate module
# layer_norm(embedding)

# # Image Example
# N, C, H, W = 20, 5, 10, 10
# input = torch.randn(N, C, H, W)
# # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# # as shown in the image below
# layer_norm = nn.LayerNorm([C, H, W])
# output = layer_norm(input)


# from huggingface_hub import hf_hub_download
# hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/config.json", local_dir="./InstantX--InstantID")
# hf_hub_download(repo_id="InstantX/InstantID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="./InstantX--InstantID")
# hf_hub_download(repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="./InstantX--InstantID")


# import requests

# def download_file(url, save_path):
#     response = requests.get(url)
#     with open(save_path, 'wb') as file:
#         file.write(response.content)

# url = 'https://images.asos-media.com/products/new-look-trench-coat-in-camel/204351106-4?$n_1920w$&wid=1926&fit=constrain'  # 要下载的URL链接
# save_path = '/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/debug/debug.jpg'

# download_file(url, save_path)


# import math
# value = float('NaN')
# print(math.isnan(value)) 


# import os 
# from tqdm import tqdm
# source_path = "/mnt/nfs/file_server/public/mingjiahui/data/TrainingDataPro--asos-e-commerce-dataset/data"
# process_dirs = [os.path.join(source_path, name) for name in os.listdir(source_path)]
# img_dirs = []
# for process_dir in tqdm(process_dirs):
#     img_dirs += [os.path.join(process_dir, name) for name in os.listdir(process_dir)]

# for img_dir in tqdm(img_dirs):
#     file_paths = [os.path.join(img_dir, name) for name in os.listdir(img_dir) if name.split('.')[1]!='jpg']
#     for file_path in file_paths:
#         save_dir = os.path.dirname(file_path)
#         suffix = os.path.basename(file_path).split('.')[1]
#         save_name =  os.path.basename(file_path).split('.')[0].zfill(6)+f'.{suffix}'
#         new_file_path = os.path.join(save_dir, save_name)
#         os.rename(file_path, new_file_path)    


# import json
# source_dir = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json"
# save_path = f"{source_dir}/trwaindata_V2_total--465217.json"
# json_list = [
#     f"{source_dir}/traindata_V1_with_all_face_info--antelopev2_non_norm--min_reso_768.json",
#     f"{source_dir}/traindata_V2_mj_xy--min_reso_768.json",
#     f"{source_dir}/traindata_V2_procucts_asos.json",
# ]
# result = []
# for json_path in json_list:
#     with open(json_path, "r")as f:
#         data = json.load(f)
#         result += data
# with open(save_path, "w")as f:
#     json.dump(result, f)
# print(f"result has saved in {save_path}")
# print(f"Total num:{len(result)}")


# import hashlib
# hash_object = hashlib.sha256()
# hash_object.update("asdfwefdfsdfgsdsfscsdfscwsf".encode())
# hashed_pair = hash_object.hexdigest()
# print(hashed_pair
#       )

# import torch
# model_file0 = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/finetune/instantid-sdxl-base/20240410-sdxl--V3--batch_64--lr1e-5--train_from_step26000/checkpoint-0/sdxl_instantid.bin"
# model_file1 = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/finetune/instantid-sdxl-base/20240410-sdxl--V3--batch_64--lr1e-5--train_from_step26000/checkpoint-0/controlnet/diffusion_pytorch_model.bin"
# sd0 = torch.load(model_file0, map_location='cpu')
# sd1 = torch.load(model_file1, map_location='cpu')
# print(sd0.keys())
# print(len(sd0['image_proj'])+len(sd0['ip_adapter']))
# print(sd0['ip_adapter'])
# print(len(sd1))
# print(sd1.keys())


# import torch
# import sys
# test = torch.randn(10000,10,10)
# test0 = test[0].clone()
# torch.save(test, "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/debug/test.bin")
# torch.save(test0, "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/debug/test0.bin")
# # print(test.shape)
# # print(sys.getsizeof(test))
# # print(test0.shape)
# # print(sys.getsizeof(test0))
# # torch.save()
# # file_path_0 = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/debug/debug0.bin"
# # # save_path = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/debug/debug1.bin"
# # # file_path_1 = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/finetune/instantid-sdxl-base/20240410-sdxl--V3--batch_64--lr1e-5--train_from_step26000/checkpoint-0/controlnet/diffusion_pytorch_model.bin"
# # file_path_2 = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/debug/debug1.bin"
# # sd0 = torch.load(file_path_0, map_location='cpu')
# # sd1 = torch.load(file_path_2, map_location='cpu')
# # print(sd0.keys() == sd1.keys())
# # print(list(sd0.keys())[0])
# # print(list(sd1.keys())[0])
# # # result = {}
# # # for k in sd0:
# # #     result[k] = sd0[k].clone
# # # torch.save(result, save_path)



##### Dataset Sampler
# import torch
# from torch.utils.data import Sampler


# class ImageSizeSampler(Sampler):
#     def __init__(self, json_mapping, image_size, batch_size):
#         self.json_mapping = json_mapping
#         self.image_size = image_size
#         self.batch_size = batch_size

#         # Create a list of indices for each image size
#         self.indices_by_size = {}
#         for idx, (image_path, _) in enumerate(json_mapping.items()):
#             size = self.get_image_size(image_path)
#             if size not in self.indices_by_size:
#                 self.indices_by_size[size] = []
#             self.indices_by_size[size].append(idx)

#         # Calculate the number of batches
#         self.num_batches = sum(len(indices) // batch_size for indices in self.indices_by_size.values())

#     def __iter__(self):
#         # Shuffle indices within each image size
#         for indices in self.indices_by_size.values():
#             torch.randperm(len(indices))

#         # Create batches
#         batches = []
#         for indices in self.indices_by_size.values():
#             for i in range(0, len(indices), self.batch_size):
#                 batches.append(indices[i:i+self.batch_size])
#         # Shuffle batches
#         torch.randperm(len(batches))

#         # Yield indices in each batch
#         for batch_indices in batches:
#             yield from batch_indices

#     def __len__(self):
#         return self.num_batches

#     def get_image_size(self, image_path):
#         # Your implementation to get the size of an image
#         return self.image_size
# # Example usage:
# # json_mapping = {"image1.jpg": {"size": [width, height]}, "image2.jpg": {"size": [width, height]}, ...}
# # image_size = [width, height]
# # batch_size = 32
# # dataset = YourDataset(json_mapping)
# # sampler = ImageSizeSampler(json_mapping, image_size, batch_size)
# # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)



# #### diffusers debug
# # from diffusers import DiffusionPipeline
# # import torch

# # pipe = DiffusionPipeline.from_pretrained("/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# # pipe.to("cuda")

# # # if using torch < 2.0
# # # pipe.enable_xformers_memory_efficient_attention()

# # prompt = "An astronaut riding a green horse"

# # images = pipe(prompt=prompt).images[0]

# from diffusers import StableDiffusionPipeline
# import torch

# model_id = "/mnt/nfs/file_server/public/mingjiahui/models/runwayml--stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]  
    
# image.save("astronaut_rides_horse.png")

# # +++++++++++++++++++++++++++++++++++++++++++


#### cos distance
# import torch
# tensor1 = torch.randn(2, 768)
# tensor2 = torch.randn(2, 768)
# l2_distance = torch.norm(tensor1 - tensor2, p=2, dim=1)
# print("L2 distance between tensor1 and tensor2:", l2_distance)

# import torch
# import numpy as np
# # tensor1 = torch.randn(1, 768)
# # tensor2 = torch.randn(1, 768)
# # numpy_tensor1 = tensor1.numpy()
# # numpy_tensor2 = tensor2.numpy()

# numpy_tensor1 = np.zeros((1, 768))
# numpy_tensor2 = np.zeros((1, 768))

# l2_distance = np.linalg.norm(numpy_tensor1 - numpy_tensor2, ord=2, axis=1)
# print("L2 distance between tensor1 and tensor2:", l2_distance)


# import numpy as np
# array1 = np.random.randn(2, 768)
# array2 = np.random.randn(2, 768)
# array1 = np.ones((2, 768))
# array2 = np.ones((2, 768))
# dot_product = np.sum(array1 * array2, axis=1)
# norm_a = np.linalg.norm(array1, axis=1)
# norm_b = np.linalg.norm(array2, axis=1)
# cosine_sim = dot_product / (norm_a * norm_b)
# print(cosine_sim)
# cosine_distance = [1 - l for l in cosine_sim]
# print("Cosine distance between array1 and array2:", cosine_distance)
# result = np.array(cosine_distance).mean()
# print(result)

# import numpy as np
# array1 = np.random.randn(2, 768)
# norm = np.linalg.norm(array1, ord=2, axis=1, keepdims=True)
# result = array1 / norm
# print(result.shape)
# print(np.linalg.norm(result[0], ord=2))


# import json
# # json_file = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/debug/if-cos.json"
# json_file = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/debug/f2d-cos.json"
# with open(json_file, 'r')as f:
#     data = json.load(f)
# print(len(data)-1)

# a = [{'a':1, 'b':2}, {'a':0, 'b':2}]
# a = sorted(a, key=lambda x: x['a'])
# print(a)


import numpy as np
npy_file = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/expression/mjh_exp-norm_embed--if_antelopev2/0_0.npy"
a = np.load(npy_file, allow_pickle=True)
print(a.shape)
print(np.linalg.norm(a, ord=2))