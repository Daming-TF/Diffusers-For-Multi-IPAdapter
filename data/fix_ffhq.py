import os
from tqdm import tqdm
data_dir = r"/mnt/nfs/file_server/public/mingjiahui/data/ffhq/data/decompression_data/in-the-wild-images/"
txt_paths = [os.path.join(data_dir, name)for name in tqdm(os.listdir(data_dir)) if name.split('.')[1] == 'txt' and os.path.exists(os.path.join(data_dir, name.replace('txt', 'png')))]
for txt_path in tqdm(txt_paths):
    with open(txt_path, 'r')as f:
        data = f.readlines()
        # assert len(data)==1, f"{txt_path}:\n{[data_ for data_ in data]}"
    prompt = data[0]
    # print(prompt)
    # print("------------")
    if '</s>' in prompt:
        prompt = prompt.replace('</s>', '')  
        # print(prompt)
        with open(txt_path, 'w')as f:
            f.write(prompt)
    #     print(f"result has saved in {txt_path}")
    # print(txt_path)
    # exit(0)