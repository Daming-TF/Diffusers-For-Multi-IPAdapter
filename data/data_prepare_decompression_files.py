import os
import tarfile
import gzip
import shutil
from tqdm import tqdm
import multiprocessing


def processing(i, tar_paths, save_dir):
    for tar_path in tqdm(tar_paths, desc=f"Processing {i}"):
        print(f"decompression file ==>{tar_path}")
        save_name = os.path.basename(tar_path).split('.')[0]
        save_path = os.path.join(save_dir, save_name)
        # if os.path.exists(save_path):
        #     continue
        os.makedirs(save_path, exist_ok=True)
        try:
            # for .tar file
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=save_path)

            # # for gzip compressed file
            # with gzip.open(tar_path, 'rb') as f_in:
            #     with open(, 'wb') as f_out:
            #         shutil.copyfileobj(f_in, f_out)
            print(f"已解压 {os.path.splitext(tar_path)} 到 {save_path}")
        except Exception as e:
            print(f"**Error: {e} ==> {tar_path}")
            continue
        


if __name__ == '__main__':
    input_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/ffhq/data/OpenDataLab___FFHQ/raw/'
    save_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/ffhq/data/decompression_data/'
    tar_paths = [os.path.join(input_dir, name) for name in os.listdir(input_dir) if 'tar' in name]

    num_process = 1
    chunk_size = len(tar_paths) // num_process
    # remain_num = len(tar_paths) % num_process
    print(f"""Total tar path num:{len(tar_paths)}\nchunk size:{chunk_size}\nprocess num:{num_process}\n""")

    processes = []
    for i in range(num_process):
        chunk = tar_paths[i*chunk_size: (i+1)*chunk_size] if i != num_process-1 else tar_paths[i*chunk_size:]
        process = multiprocessing.Process(target=processing, args=(i, chunk, save_dir))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
    
