import json
import os
from tqdm import tqdm
import multiprocessing
import sys


data_dict = {
        'Laion': ['data-50m_arcface', 'data-50m_antelopev2'],
        'coyo': ['coyo700m/data_arcface', 'coyo700m/data_antelopev2'],
        'ffhq': ['in-the-wild-images_arcface', 'in-the-wild-images_antelopev2'],
}


def processor(i, data, save_dir):
    save_path = os.path.join(save_dir, f"{i}.json")
    result = []
    for metadata in tqdm(data):
        image_path = metadata['image_file']
        embeds_path = metadata['embeds_path']
        find_status = False
        for key in data_dict.keys():
            if key in image_path:
                find_status = True
                break
        assert find_status, ValueError(f"some error happened\t==>{image_path}")
        embeds_path = embeds_path.replace(data_dict[key][0], data_dict[key][1])
        if os.path.exists(embeds_path):
            metadata['embeds_path'] = embeds_path
            result.append(metadata)
    with open(save_path, 'w')as f:
        json.dump(result, f)
        print(f"result has saved in {save_path}")


if __name__ == '__main__':
    json_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_crop.json"
    save_dir = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/_tmp/antelopev2_concat_json"
    os.makedirs(save_dir, exist_ok=True)
    save_path = "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_crop_antelopev2.json"
    with open(json_path, 'r')as f:
        data = json.load(f)

    process_num = 8
    chunk_num = len(data) // process_num
    residue_num = len(data) % process_num
    process_list = []
    data_index = 0
    for i in range(process_num):
        if i < residue_num:
            chunk_input = data[data_index: data_index+chunk_num+1]
            data_index += chunk_num+1
        else:
            chunk_input = data[data_index: data_index+chunk_num]
            data_index += chunk_num
        
        process = multiprocessing.Process(target=processor, args=(i, chunk_input, save_dir))
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()

    result = []
    for i in range(process_num):
        json_path = os.path.join(save_dir, f"{i}.json")
        with open(json_path, 'r')as f:
            data = json.load(f)
            result += data
    with open(save_path, 'w')as f:
        json.dump(result, f)
        print(f"total data:{len(result)}")
        print(f"result has saved in {save_path}")