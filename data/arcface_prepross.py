import argparse
import multiprocessing
from multiprocessing import Queue
import threading
from tqdm import tqdm
from insightface.app import FaceAnalysis
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import json
import os
import sys
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(current_path))
from my_script.util.util import FaceidAcquirer, image_grid

import logging
from logging.handlers import QueueHandler, QueueListener
logging.basicConfig(level=logging.WARNING)
log_queue = Queue()
handler = QueueHandler(log_queue)
logger = logging.getLogger()
logger.addHandler(handler)
listener = QueueListener(log_queue, \
            logging.FileHandler('/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/data/_tmp/logfile.log'))
listener.start()


def _get_image_paths(image_dirs):
    result = {}
    threading_num = 4
    chunk_num = len(image_dirs) // threading_num
    residue_num = len(image_dirs) % threading_num

    def __get_image_paths(index, image_dirs):
        image_paths = []
        for image_dir in tqdm(image_dirs):
            image_paths += [os.path.join(image_dir, name) for name in os.listdir(image_dir) if name.split('.')[1] in ['jpg','png']]
        result[index] = image_paths

    process_list = []
    data_index = 0
    for i in range(threading_num):
        if i < residue_num:
            chunk_input = image_dirs[data_index: data_index+chunk_num+1]
            data_index += chunk_num+1
        else:
            chunk_input = image_dirs[data_index: data_index+chunk_num]
            data_index += chunk_num
        thread_process = threading.Thread(target=__get_image_paths, args=[i, chunk_input])
        thread_process.start()
        process_list.append(thread_process)
    
    for process in process_list:
        process.join()
    
    return result


def get_image_paths(input_dir, data_type):
    image_dirs = []
    if data_type in ['laion', 'imdb']:
        input_paths = [os.path.join(input_dir, name) for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]
        for input_path in tqdm(input_paths):
            image_dirs += [os.path.join(input_path, name) for name in os.listdir(input_path)]
    elif data_type in ['ffhq','debug']:
        image_dirs = [args.input]
        # print(f"**Debug:{image_dirs}")
    else: 
        ValueError("only support laion data currently")

    num_process = 1 if data_type in ['debug', 'ffhq'] else 4
    chunk_num = len(image_dirs) // num_process
    residue_num = len(image_dirs) % num_process

    multi_process_input = []
    data_index = 0
    for i in range(num_process):
        if i < residue_num:
            chunk_input = image_dirs[data_index: data_index+chunk_num+1]
            data_index = data_index+chunk_num+1
        else: 
            chunk_input = image_dirs[data_index: data_index+chunk_num]
            data_index = data_index+chunk_num
        multi_process_input.append(chunk_input)
        
    with multiprocessing.Pool(processes=num_process) as pool:
        results = pool.map(_get_image_paths, multi_process_input) 
        pool.close()
        pool.join()
    
    print(f"result:{type(results)}") 
    print(len(results))      # length is equivalent to threading  num
    print(type(results[0]))
    print(results[0].keys())
    print(type(results[0][0]))
    print(len(results[0][0]))
    image_paths = []
    for result_multiprocess in results:
        for thread_index, thread_image_paths in result_multiprocess.items():
            image_paths += thread_image_paths
    image_paths = sorted(image_paths)

    return image_paths


def processing(process_index, image_paths, output_dir, data_type):
    if data_type not in  ['laion', 'coyo','ffhq', 'imdb']:
        ValueError("'data_type' only support for ['laion', 'coyo', 'ffhq']  currently")

    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    transform = transforms.Resize(1024)
    
    progress_bar = tqdm(image_paths) if process_index==0 or process_index==3 else image_paths
    for index, image_path in enumerate(progress_bar):
        try:
            image = transform(Image.open(image_path).convert("RGB"))
            image = np.array(image)[:, :, ::-1]

            image_name = os.path.basename(image_path)
            suffix = image_name.split('.')[1]
            image_dir = os.path.dirname(image_path)
            
            bbox = None
            embeds_path = None
            faces = app.get(image)
            if len(faces) == 0:
                continue
            else:
                if data_type in ['laion', 'imdb']:
                    dir_name = os.path.basename(image_dir)
                    input_name = os.path.basename(os.path.dirname(image_dir))
                    save_dir = os.path.join(output_dir, input_name, dir_name)
                elif data_type=='coyo':
                    dir_name = os.path.basename(image_dir)
                    save_dir = os.path.join(output_dir, dir_name)
                elif data_type=='ffhq':
                    save_dir = output_dir
                
                os.makedirs(save_dir, exist_ok=True)
                json_path = os.path.join(save_dir, image_name.replace(suffix, 'json'))
                embeds_path = json_path.replace('json', 'npy')

                if len(faces) ==1:
                    # get embeds
                    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
                    bbox = faces[0]['bbox'].tolist()
                    np.save(embeds_path, faceid_embeds)

            # get txt prmpt
            txt = None
            txt_path = image_path.replace(suffix, 'txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r')as f:
                    txt_lines = f.readlines()
                    assert len(txt_lines)==1, "txt_lines not only one lines data"
                    txt = txt_lines[0]
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"**Error:{e} ==> {image_path}\t{txt_path}")
            continue
        
        metadata = {
            "image_path":image_path,
            "bbox":bbox,
            "embeds_path":embeds_path,
            "txt":txt,
            "ori_size":image.size,
            "detect_reso":1024,
            "status":1 if len(faces) ==1 else 2,
        }

        with open(json_path, 'w')as f:
            json.dump(metadata, f)
        if index % 100 == 0 and (process_index==0 or process_index==3):
            print(f"result has saved in {json_path}")


def get_image_paths_byjson(json_path):
    image_paths = []
    with open(json_path, 'r')as f:
        data = json.load(f)
        assert isinstance(data, list)
    for meta_data in data:
        image_paths.append(meta_data['image_file'])
    image_paths = sorted(image_paths)
    return image_paths


def remove_paths(arg_tuple):
    process_index, image_paths, output_dir, data_type = arg_tuple
    result = []
    progress_bar = tqdm(image_paths) if process_index==0 else image_paths
    for image_path in progress_bar:
        image_name = os.path.basename(image_path)
        suffix = image_name.split('.')[1]
        image_dir = os.path.dirname(image_path)

        if data_type in ['laion', 'imdb']:
            dir_name = os.path.basename(image_dir)
            input_name = os.path.basename(os.path.dirname(image_dir))
            save_dir = os.path.join(output_dir, input_name, dir_name)
        elif data_type=='coyo':
            dir_name = os.path.basename(image_dir)
            save_dir = os.path.join(output_dir, dir_name)
        elif data_type=='ffhq':
            save_dir = output_dir
        else:
            ValueError(f"data_type is only support ['laion','coyo','ffhq','imdb']")
        json_path = os.path.join(save_dir, image_name.replace(suffix, 'json'))
        if not os.path.exists(json_path):
            result.append(image_path)
        # else:
        #     print(f"{json_path} has exists!") if process_index==0 else None
    return result


def main(args):
    # 1.get image paths
    if args.input_json is None:
        image_paths = get_image_paths(args.input, data_type=args.datatype)
    else:
        image_paths = get_image_paths_byjson(args.input_json)
    print(f"\033[91m  Total data:{len(image_paths)}  \033[0m")

    # 2.Remove the processed file path
    if args.remove:
        multi_process_num = 4*4
        chunk_num = len(image_paths) // multi_process_num
        residue_num = len(image_paths) % multi_process_num
        multi_process_input = []
        data_index = 0
        for i in range(multi_process_num):
            if i < residue_num:
                chunk_input = image_paths[data_index: data_index+chunk_num+1]
                data_index = data_index+chunk_num+1
            else: 
                chunk_input = image_paths[data_index: data_index+chunk_num]
                data_index = data_index+chunk_num
            multi_process_input.append((i, chunk_input, args.output, args.datatype))
            
        with multiprocessing.Pool(processes=multi_process_num) as pool:
            results = pool.map(remove_paths, multi_process_input) 
            pool.close()
            pool.join()
        
        new_image_paths = []
        for result in results:
            new_image_paths += result
        image_paths = sorted(new_image_paths)


    # 3.multi process
    multi_process_num = 4
    chunk_num = len(image_paths) // multi_process_num
    residue_num = len(image_paths) % multi_process_num

    process_list = []
    data_index = 0
    for i in range(multi_process_num):
        if i < residue_num:
            chunk_input = image_paths[data_index: data_index+chunk_num+1]
            data_index += chunk_num+1
        else:
            chunk_input = image_paths[data_index: data_index+chunk_num]
            data_index += chunk_num

        process = multiprocessing.Process(target=processing, args=(i, chunk_input, args.output, args.datatype))
        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()



if __name__ == '__main__':
    data_dict = {
        'laion':{
            'input':'/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/data/data-50m',
            'output':'/mnt/nfs/file_server/public/mingjiahui/data/Laion400m_face/data/data-50m_arcface',
        },
        'debug':{
            'input':'/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data',
            'output':'/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/all_test_data_arcface'
        },
        'coyo':{
            'input':'/mnt/nfs/file_server/public/mingjiahui/data/coyo700m/data',
            'output':'/mnt/nfs/file_server/public/mingjiahui/data/coyo700m/data_arcface'
        },
        'ffhq':{
            'input':'/mnt/nfs/file_server/public/mingjiahui/data/ffhq/data/decompression_data/in-the-wild-images',
            'output':'/mnt/nfs/file_server/public/mingjiahui/data/ffhq/data/decompression_data/in-the-wild-images_arcface'
        },
        'imdb':{
            'input':'/mnt/nfs/file_server/public/mingjiahui/data/imdb-wiki/data',
            'output':'/mnt/nfs/file_server/public/mingjiahui/data/imdb-wiki/data_arcface'
        },
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    # parser.add_argument("--debug", action='store_true')
    parser.add_argument("--datatype", type=str, \
                        help="Union['laion', 'debug','ffhq','coyo']")
    parser.add_argument("--remove", action='store_true')
    parser.add_argument("--input_json", type=str, default=None)
    args = parser.parse_args()
    assert args.datatype in ['laion', 'debug', 'coyo', 'ffhq', 'imdb']
    args.input = data_dict[args.datatype]['input']
    args.output = data_dict[args.datatype]['output']
    main(args)