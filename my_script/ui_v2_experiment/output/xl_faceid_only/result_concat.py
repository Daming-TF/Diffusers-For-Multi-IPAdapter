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
    compare_class_list = ['ori', 'sd15-faceid-plusv2', 'xl-faceid']
    lora_id = "graffitisplash_sd"
    test_data_id = "all_test_data"
    ori_input = fr'./data/{test_data_id}'
    sd15_plusv2_results_input = fr'./data/{test_data_id}_lora_script_output/{lora_id}'
    xl_results_input = fr"./my_script/ui_v2_experiment/output/{test_data_id}/{lora_id}"
    save_path = fr"/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/other/compare_sd15plusv2_and_faceidxl--lora-{lora_id}--{test_data_id.replace('.','_')}.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 2. prepare data
    # sd15_plusv2_results = []
    sd15_plusv2_results = [os.path.join(sd15_plusv2_results_input, name) for name in os.listdir(sd15_plusv2_results_input)]
    # for sd15_plusv2_results_dir in sd15_plusv2_results_dirs:
    #     sd15_plusv2_results += [os.path.join(sd15_plusv2_results_dir, name) for name in os.listdir(sd15_plusv2_results_dir)]
        
    # 3. prepare black_bar 
    result = np.zeros((200, target_size*(len(compare_class_list)+1), 3), dtype=np.uint8)
    for i, class_name in enumerate(compare_class_list):
        local = 20+target_size*i if i == 0 else 20+target_size*(i+1)
        cv2.putText(result, class_name, (local, 120), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1,
                    color=(255,255,255),
                    thickness=4
                    )    
    Image.fromarray(result).save(r'/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/other/debug.jpg')

    # 4. processing
    for sd15_plusv2_image_path in tqdm(sd15_plusv2_results):
        image_name = os.path.basename(sd15_plusv2_image_path)
        ori_path = os.path.join(ori_input, image_name)
        # lora_name = os.path.basename(os.path.dirname(sd15_plusv2_image_path))
        xl_face_id_image_path = os.path.join(xl_results_input, image_name)
        # print(f'debug:{xl_face_id_image_path}')
        
        if not os.path.exists(xl_face_id_image_path):
            continue
        
        # print(f"image0 ==> {ori_path}")
        # print(f"image1 ==> {sd15_plusv2_image_path}")
        # print(f"image2 ==> {xl_face_id_image_path}")
        # print(f"save_path ==> {save_path}")
        # exit(0)

        ori_image = np.array(resize_and_fill(ori_path, (target_size*2, target_size*2)).convert("RGB"))
        sd15_plusv2_result = Image.open(sd15_plusv2_image_path).resize((target_size, target_size*2)).convert("RGB")
        xl_result = Image.open(xl_face_id_image_path).resize((target_size, target_size*2)).convert("RGB")

        cv2.putText(ori_image, image_name.split('.')[0], (40,40), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(255,0,0),
                    thickness=2)
        concat = cv2.hconcat([np.array(ori_image), np.array(sd15_plusv2_result), np.array(xl_result)])
        result = cv2.vconcat([result, concat]) if result is not None else concat

    Image.fromarray(result).save(save_path)
    print(f"result has saving in {save_path}")


if __name__ == '__main__':
    main()