from insightface.app import FaceAnalysis
import numpy as np
import cv2
from PIL import Image
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str, required=True)
    args = parser.parse_args()
    
    img_dirs = [
        "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/average_id",
        "/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/famous"
    ]
    img_paths = []
    for img_dir in img_dirs:
        img_paths += [os.path.join(img_dir, name) for name in os.listdir(img_dir) \
                      if name.split('.')[-1] not in ['txt', 'npy']]
    
    result_img_paths = [os.path.join(args.input, name) for name in os.listdir(args.input) if name.split('.')[-1] not in ['txt', 'npy']]

    
        
    
