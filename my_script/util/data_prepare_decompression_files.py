import os
import tarfile
from tqdm import tqdm
import multiprocessing


def processing(tar_paths):
    for tar_path in tqdm(tar_paths, desc=f"Processing {tar_paths[0]}"):
        folder_path = tar_path.split('.')[0]
        if os.path.exists(folder_path):
            continue
        os.makedirs(folder_path, exist_ok=True)
        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=folder_path)
            print(f"已解压 {os.path.splitext(tar_path)} 到 {folder_path}")
        except Exception as e:
            print(f"**Error: {e} ==> {tar_path}")
            continue
        


if __name__ == '__main__':
    input_dir = r'/mnt/nfs/file_server/public/mingjiahui/data/coyo700m/data/'
    tar_paths = [os.path.join(input_dir, name) for name in os.listdir(input_dir) if name.endswith('.tar')]

    num_process = 60
    chunk_size = len(tar_paths) // num_process
    path_chunks = []
    print(
        f"""
        Total tar path num:{len(tar_paths)}\n
        chunk size:{chunk_size}\n
        process num:{num_process}\n
        """
          )
    for i in range(num_process):
        chunk = tar_paths[i*chunk_size: (i+1)*chunk_size] if i != num_process-1 else tar_paths[i*chunk_size:]
        path_chunks.append(chunk)

    pool = multiprocessing.Pool(processes=num_process)
    pool.map(processing, path_chunks)
    pool.close()
    pool.join()
    
