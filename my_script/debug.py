from safetensors import torch
path0 = r'/mnt/nfs/file_server/public/mingjiahui/models/h94--IP-Adapter/faceid/ip-adapter-faceid_sd15_lora.safetensors'
path1 = r'/mnt/nfs/file_server/public/mingjiahui/models/lora_1.5/Joyful_Cartoon.safetensors'
sd0 = torch.load_file(path0)
sd0_keys= [k for k in sd0.keys() if 'unet' in k]
sd1 = torch.load_file(path1)
sd1_keys= [k for k in sd1.keys() if 'unet' in k]

print(sd0.keys())
print("---------------")
print(sd1.keys())
print(set(sd0.keys())==set(sd1.keys()))
