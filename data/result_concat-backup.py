from PIL import Image
import cv2
import os
import numpy as np
from tqdm import tqdm

# def resize_image(image_path, new_size):
#     original_image = Image.open(image_path)
#     old_size = original_image.size
#     new_image = Image.new("RGB", new_size, (0, 0, 0))
#     position = ((new_size[0] - old_size[0]) // 2, (new_size[1] - old_size[1]) // 2)
#     new_image.paste(original_image, position)
#     return new_image

def resize_and_fill(image_path, new_size):
    original_image = Image.open(image_path)
    old_width, old_height = original_image.size

    if old_width > old_height:
        new_width = min(new_size)
        new_height = new_width*old_height/old_width
    else:
        new_height = min(new_size)
        new_width = new_height*old_width/old_height

    resized_image = original_image.resize((int(new_width), int(new_height)))
    new_image = Image.new("RGB", new_size, (0, 0, 0))

    paste_position = (int((new_size[0] - new_width) // 2), int((new_size[1] - new_height) // 2))
    new_image.paste(resized_image, paste_position)

    return new_image


def main():
    # 1. set input path
    target_size = 512
    compare_class_list = ['ori', 'sdxl-faceid-plusv2', 'InstantID']
    ori = r'./data/all_test_data'
    ip_faceid = r'./data/all_test_data_xl_faceid_plus_v2'
    instantID = r'./InstantID/script_output'
    save_path = r'/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/treated_result/compare-ip_v2-InstantID/result.jpg'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 2. prepare data
    ori_image_paths = [os.path.join(ori, name) for name in os.listdir(ori) if name.endswith('.jpg')]
  
    # 3. prepare black_bar 
    result = np.zeros((200, target_size*len(compare_class_list), 3), dtype=np.uint8)
    for i, class_name in enumerate(compare_class_list):
        cv2.putText(result, class_name, (20+target_size*i, 120), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1,
                    color=(255,255,255),
                    thickness=4
                    )    
    Image.fromarray(result).save(r'/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/other/debug.jpg')

    # 4. processing
    for image_path in tqdm(ori_image_paths):
        image_name = os.path.basename(image_path)
        ip_faceid_path = os.path.join(ip_faceid, image_name)
        instantID_path = os.path.join(instantID, 'ink_'+image_name)

        if not os.path.exists(ip_faceid_path) or not os.path.exists(instantID_path):
            print(f"ip_faceid_path:{ip_faceid_path}\ninstantID_path:{instantID_path}")
            continue

        ip_faceid_image = Image.open(ip_faceid_path).resize((target_size, target_size))
        instantID_image = Image.open(instantID_path).resize((target_size, target_size))
        ori_image = np.array(resize_and_fill(image_path.replace('', ''), (target_size, target_size)))
        # cv2.putText(ori_image, os.path.basename(image_path).split('.')[0], (40,40), 
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
        #             color=(255,0,0),
        #             thickness=2)
        concat = cv2.hconcat([np.array(ori_image), np.array(ip_faceid_image), np.array(instantID_image)])
        result = cv2.vconcat([result, concat]) if result is not None else concat

    Image.fromarray(result).save(save_path)
    print(f"result has saving in {save_path}")


if __name__ == '__main__':
    main()