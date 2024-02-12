import json
source_dir = "/mnt/nfs/file_server/public/mingjiahui/data"
# data_dict = {
#     'laion': f"{source_dir}/Laion400m_face/data/_tmp/train_V1-laion_all_one_face.json",
#     'coyo': f"{source_dir}/coyo700m/_tmp/train_V1-coyo_all_one_face.json",
#     'ffhq': f"{source_dir}/ffhq/data/decompression_data/_tmp/train_V1-ffhq_all_one_face.json",
# }
data_dict = {
    'laion': f"{source_dir}/Laion400m_face/data/_tmp/train_instantid_controlnet-laion_all_one_face.json",
    'coyo': f"{source_dir}/coyo700m/_tmp/train_instantid_controlnet-coyo_all_one_face.json",
    'ffhq': f"{source_dir}/ffhq/data/decompression_data/_tmp/train_instantid_controlnet-ffhq_all_one_face.json",
}
# save_path = '/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/tran_json/traindata_V1.json'
save_path = '/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/trainInstantID_V1.json'
data_list = []
for data_type, json_path in data_dict.items():
    with open(json_path) as f:
        data = json.load(f)
        print(f"{data_type}: {len(data)}")
        data_list += data

print(f"Total num :{len(data_list)}")
with open(save_path, 'w')as f:
    json.dump(data_list, f) 
print(f"reuslt has saved in {save_path}")