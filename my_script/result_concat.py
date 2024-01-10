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
        new_width = new_size[0]
        new_height = new_width*old_height/old_width
    else:
        new_height = new_size[1]
        new_width = new_height*old_width/old_height

    resized_image = original_image.resize((int(new_width), int(new_height)))
    new_image = Image.new("RGB", new_size, (0, 0, 0))

    paste_position = (int((new_size[0] - new_width) // 2), int((new_size[1] - new_height) // 2))
    new_image.paste(resized_image, paste_position)

    return new_image


def main():
    # 1. set input path
    target_size = 512
    compare_class_list = ['ori', 'sd15-faceid-plusv2', 'xl-faceid', 'xl-faceid']
    xl_results_0 = r'/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/test_data_V2_xl_script_output'
    xl_results_1 = r'/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/test_data_V2_xl_script_scale0.6-ip_lora0.6_output'
    sd15_plusv2_results = r'/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/test_data_V2_sd15_plusv2_script_output'
    save_path = r'/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/other/compare_sd15plusv2_and_faceidxl.jpg'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 2. prepare data
    image_dirs = [os.path.join(sd15_plusv2_results, name) for name in os.listdir(sd15_plusv2_results)]
    image_paths = []
    for image_dir in image_dirs:
        image_paths += [os.path.join(image_dir, name) for name in os.listdir(image_dir) \
                        if name.split('.')[1] in ['jpg', 'png', 'webp']]
        
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
    for image_path in tqdm(image_paths):
        sd15_plusv2_result = Image.open(image_path).resize((target_size, target_size))
        xl_result_0 = Image.open(image_path.replace(os.path.basename(sd15_plusv2_results), \
                                                    os.path.basename(xl_results_0))).resize((target_size, target_size))
        xl_result_1 = Image.open(image_path.replace(os.path.basename(sd15_plusv2_results), \
                                                    os.path.basename(xl_results_1))).resize((target_size, target_size))
        ori_image = np.array(resize_and_fill(image_path.replace('_sd15_plusv2_script_output', ''), (target_size, target_size)))
        cv2.putText(ori_image, os.path.basename(image_path).split('.')[0], (40,40), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(255,0,0),
                    thickness=2)
        concat = cv2.hconcat([np.array(ori_image), np.array(sd15_plusv2_result), np.array(xl_result_0), np.array(xl_result_1)])
        result = cv2.vconcat([result, concat]) if result is not None else concat

    Image.fromarray(result).save(save_path)
    print(f"result has saving in {save_path}")


if __name__ == '__main__':
    main()