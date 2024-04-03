from deepface import DeepFace
import os
import cv2
from tqdm import tqdm
import pandas as pd
import sys
import logging
current_path = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from data.xlsx_writer import WriteExcel


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    logging.basicConfig(level=logging.ERROR)
    print(os.environ['TF_FORCE_GPU_ALLOW_GROWTH'])
    models = [
    "VGG-Face", 
    "Facenet", 
    "Facenet512", 
    "OpenFace", 
    "DeepFace", 
    "DeepID", 
    "ArcFace", 
    "Dlib", 
    "SFace",
    ]
    # init
    result = {}
    test0_data_dir = "./data/all_test_data/"
    test0_data_paths = [os.path.join(test0_data_dir, name)for name in os.listdir(test0_data_dir)\
                        if not name.endswith('.txt')]
    test1_data_dir = "./data/test_data_V2/"
    test1_data_dirs_ = [os.path.join(test1_data_dir, dir_name) for dir_name in os.listdir(test1_data_dir)]
    print(test1_data_dirs_)
    test1_data_paths = []
    # for test1_data_dir_ in test1_data_dirs_:
    #     test1_data_paths += [os.path.join(test1_data_dir_, name)for name in os.listdir(test1_data_dir_)\
    #                         if not name.endswith('.txt')]
    test_data_paths = test0_data_paths + test1_data_paths
    print(f"test0 num:{len(test0_data_paths)}\ttest1 num:{len(test1_data_paths)}\ttotal num:{len(test_data_paths)}")
    target_img= "./data/all_test_data/wang1.jpg"
    save_path = "./data/other/distance.xlsx"
    test_data_ids = [os.path.basename(test_data_path).split('.')[0] \
                    for test_data_path in test_data_paths]
    test_models = [
        "Facenet512", 
        "SFace", 
        "ArcFace", 
        "VGG-Face",
    ]

    xlsx_writer = WriteExcel(save_path, test_data_ids, test_models)

    # process
    total_result = {}
    for i, model in enumerate(test_models):
        col_result = []
        for test_data_path in tqdm(test_data_paths):
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            result_ = DeepFace.verify(img1_path = target_img, 
                img2_path = test_data_path, 
                model_name=model,
                detector_backend="mtcnn"
            )['distance']
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"{os.path.basename(target_img)}\t{os.path.basename(test_data_path)}\t{result_}") 
            col_result.append(result_)
        xlsx_writer.write(col_result, i)
        # total_result[model] = col_result
    xlsx_writer.close()
    print(f"result has saved in {save_path}")



