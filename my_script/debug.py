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

import torch
# from safetensors.torch import load_file
# model_path = "/mnt/nfs/file_server/public/mingjiahui/models/lllyasviel--control_v11p_sd15_openpose/diffusion_pytorch_model.fp16.bin"
# sd0 = torch.load(model_path)

# ours_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/finetune/instantid/20240211-sd15--V1-wo_xformer-scratch/checkpoint-1/diffusion_pytorch_model.safetensors"
# sd1 = load_file(ours_path, device="cpu")
# print(set(sd0) == set(sd1))

tensor = torch.randn(3, 4, 5)
print(tensor.size())
print(tensor.size(0))
print(tensor.shape)