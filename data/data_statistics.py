import os
import argparse
import numpy as np
import threading
import multiprocessing
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


def threading_processor(json_dirs, result_dict:dict, i):
    result_dict.setdefault(i, {})
    reso_np = np.arange(512, 1024+128, 128)     # [512,640,768,896,1024]
    for reso in reso_np.tolist():
        result_dict[i].setdefault(reso, [])
    result_dict[i].setdefault('surpass_1024', [])

    single_face_reso_list = [0]*(reso_np.shape[0]+1)        # [0, 0, 0, 0, 0, 0]
    multi_face_reso_list = [0]*(reso_np.shape[0]+1)        # [0, 0, 0, 0, 0, 0]
    
    
    progress_bar = tqdm(json_dirs, disable=False) if i==0 else tqdm(json_dirs, disable=True)
    json_paths = []
    for json_dir in progress_bar:
        json_paths += [os.path.join(json_dir, name) for name in os.listdir(json_dir) if name.endswith('.json')]
    
    for json_path in tqdm(json_paths):
        try:
            with open(json_path, 'r')as f:
                meta_data = json.load(f)
        
            image_path = meta_data['image_path']
            ori_reso = meta_data['ori_size']
            state = meta_data['status']

            find = False
            for reso_index, reso in enumerate(reso_np):
                if min(ori_reso) < reso:
                    find = True
                    if state == 1:
                        single_face_reso_list[reso_index] += 1
                    else: 
                        multi_face_reso_list[reso_index] += 1
                    break
            if not find:
                if state == 1:
                    single_face_reso_list[-1] += 1
                else: 
                    multi_face_reso_list[-1] += 1
                result_dict[i]['surpass_1024'].append(image_path)
            else:
                result_dict[i][reso].append(image_path)
        except Exception as e:
            print(f"{e}\t==>\t{json_path}")
            exit(0)
            # continue
    
    result_dict[i]['single'] = single_face_reso_list
    result_dict[i]['multi'] = multi_face_reso_list


def main(args):
    # get json_dir
    assert args.data in ['coyo', 'laion', 'ffhq', 'imdb'], "only support data for ['coyo', 'laion', 'ffhq', 'imdb']"
    if args.data in ['coyo']:
        json_dirs = [os.path.join(args.input, name) for name in tqdm(os.listdir(args.input)) if os.path.isdir(os.path.join(args.input, name))]
    elif args.data in ['laion','imdb']:
        file_dirs = [os.path.join(args.input, name) for name in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, name))]
        file_dirs = file_dirs[:1] if args.debug is not None else file_dirs
        json_dirs = []
        for file_dir in tqdm(file_dirs):
            json_dirs += [os.path.join(file_dir, name)for name in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, name))]
    elif args.data in ['ffhq']:
        json_dirs = [args.input]
    else:
        ValueError("data type only support for ['coyo', 'laion']")

    chunk_num = len(json_dirs) // args.process_num
    residue_num = len(json_dirs) % args.process_num
    print(f"chunk num:{chunk_num}\tresidue num:{residue_num}")

    # Start multithreading
    result_dict = {}
    data_index = 0
    threads = []
    if args.debug:
        threading_processor(json_dirs[:10], result_dict, 0)
    else:
        for i in range(args.process_num):
            if i < residue_num:
                chunk_dirs = json_dirs[data_index:data_index+chunk_num+1]
                data_index = data_index+chunk_num+1
            else:
                chunk_dirs = json_dirs[data_index:data_index+chunk_num]
                data_index = data_index+chunk_num
            thread = threading.Thread(target=threading_processor, args=(chunk_dirs, result_dict, i))
            # thread = multiprocessing.Process(target=threading_processor, args=(chunk_dirs, result_dict, i))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
    
    # Summarize the returns of all threads
    # from collections import OrderedDict
    # result_dict = OrderedDict(sorted(result_dict.items()))
    total_result = {}
    for _, meta_data in result_dict.items():
        for k, v in meta_data.items():
            total_result.setdefault(k, [])
            if k not in ['single', 'multi']:
                total_result[k] += v  
            else:
                total_result[k] = (np.array(total_result[k]) + np.array(v)).tolist() if len(total_result[k]) != 0 else np.array(v)

    # save_json
    save_path = os.path.join(args.output, f'{args.data}.json')
    with open(save_path, 'w')as f:
        json.dump(total_result, f)

    total_num = 0
    for num_0, num_1 in zip(total_result['single'], total_result['multi']):
        total_num += num_0+num_1
    print(f"\033[91m Total num: {total_num}\t ==> \tSingle:{total_result['single']}\tMulti:{total_result['multi']} \033[0m")

    # save_statistics_result
    image_save_path = os.path.join(args.output, f'{args.data}.jpg')   
    single_face_reso_list=total_result['single']
    multi_face_reso_list=total_result['multi']
    x_labels = ['<512', '512-640', '640-768', '768-896', '896-1024', '>1024']
    x = np.arange(len(x_labels))
    plt.bar(x-0.2, single_face_reso_list, width=0.4, color='blue')
    plt.bar(x+0.2, multi_face_reso_list, width=0.4, color='red')
    for i, v in enumerate(single_face_reso_list):
        plt.text(i - 0.2, v + 1, str(v), ha='center', va='bottom')
    for i, v in enumerate(multi_face_reso_list):
        plt.text(i + 0.2, v + 1, str(v), ha='center', va='bottom')
    plt.xticks(x, x_labels)
    plt.savefig(image_save_path)
    print(f"result save in {image_save_path}")

    # transfer statistics json to training json
    # target reso > 896
    if args.transfer_to_train:
        result = []
        with open(save_path, 'r')as f:
            data = json.load(f)
        for reso, image_paths in tqdm(data.items()):
            if reso not in ['512', '640', '768', '896', '1024', 'surpass_1024']:
                continue
            for image_path in tqdm(image_paths):
                meta_data = {}

                suffix = os.path.basename(image_path).split('.')[1]
                if args.data == 'coyo':
                    json_path = image_path.replace('coyo700m/data', 'coyo700m/data_arcface').replace(suffix, 'json')
                elif args.data == 'laion':
                    json_path = image_path.replace('data-50m', 'data-50m_arcface').replace(suffix, 'json')
                elif args.data == 'ffhq':
                    json_path = image_path.replace('in-the-wild-images', 'in-the-wild-images_arcface').replace(suffix, 'json')
                with open(json_path, 'r') as f:
                    metadate = json.load(f)
                    txt = metadate['txt']
                    status = metadate['status']
                    embeds_path = json_path.replace('json', 'npy')
                
                if status!=1:
                    continue
                assert os.path.exists(embeds_path), f"{json_path} <==> {embeds_path}"

                meta_data["image_file"] = image_path
                meta_data["text"] = txt
                meta_data["embeds_path"] = embeds_path
                result.append(meta_data)
        
        print("saving the json.......")
        train_json_path = os.path.join(args.output, f'train_V2-{args.data}_all_one_face.json')
        with open(train_json_path, 'w') as f:
            json.dump(result, f)
        print(f"Total data:{len(result)}")
        print(f"result save in {train_json_path}")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    # parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--data", type=str, required=True,help="union['coyo','laion','ffhq','imdb']")
    parser.add_argument("--process_num", type=int, default=16)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--transfer_to_train", action='store_true')
    args = parser.parse_args()
    data_dict = {
        'coyo':{
            'input':'/mnt/nfs/file_server/public/mingjiahui/data/coyo700m/data_arcface',
            'output':'/mnt/nfs/file_server/public/mingjiahui/data/coyo700m/_tmp',
        },
        'laion':{
            'input':'/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/data/data-50m_arcface/',
            'output':'/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/data/_tmp',
        },
        'ffhq':{
            'input':'/mnt/nfs/file_server/public/mingjiahui/data/ffhq/data/decompression_data/in-the-wild-images_arcface',
            'output':'/mnt/nfs/file_server/public/mingjiahui/data/ffhq/data/decompression_data/_tmp'
        },
        'imdb':{
            'input':'/mnt/nfs/file_server/public/mingjiahui/data/imdb-wiki/data_arcface',
            'output':'/mnt/nfs/file_server/public/mingjiahui/data/imdb-wiki/_tmp'
        },
    }
    args.input = data_dict[args.data]['input'] if args.input is None else args.input
    args.output = data_dict[args.data]['output'] if args.output is None else args.output
    os.makedirs(args.output, exist_ok=True)
    main(args)