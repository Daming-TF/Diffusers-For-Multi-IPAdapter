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


def resize_and_fill(image_path, new_size:tuple) -> Image.Image:
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


def pic_frame(image_path, txt, size:tuple) -> Image.Image:
    image = Image.open(image_path).resize((size[0]-size[0]//40, size[1]-size[1]//40))
    old_width, old_height = image.size
    new_image = Image.new("RGB", size, (0, 0, 0))
    paste_position = (int((size[0]-old_width) // 2), int((size[1]-old_height) // 2))
    new_image.paste(image, paste_position)
    new_image = cv2.putText(np.array(new_image), txt, (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0,0,255), thickness=2)
    return new_image


def main():
    # 1. set input path
    target_size = 1024
    # ['3dpixel2_sd', 'papercutout_sd', '3dexaggeration_sd', 'graffitisplash_sd', 'holographic_sd']
    lora_id = "holographic_sd"
    test_data_id = "all_test_data"
    ori_input = fr'./data/{test_data_id}'
    sd15_plusv2_results_input = fr'./data/{test_data_id}_lora_script_output/{lora_id}'

    xl_results_dir = fr"./my_script/ui_v2_experiment/output/diff_pipe_comparison/{test_data_id}/{lora_id}"
    save_path = fr"/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/other/compare_diff_pipe--lora-{lora_id}--{test_data_id.replace('.','_')}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 2. prepare data
    sd15_plusv2_results = [os.path.join(sd15_plusv2_results_input, name) for name in os.listdir(sd15_plusv2_results_input)]

    # 4. processing
    result = None
    for sd15_plusv2_image_path in tqdm(sd15_plusv2_results):
        image_name = os.path.basename(sd15_plusv2_image_path)
        ori_path = os.path.join(ori_input, image_name)

        pipe0_Path = os.path.join(xl_results_dir, "only_faceid_0.6", image_name)
        pipe1_Path = os.path.join(xl_results_dir, "only_faceid_0.7", image_name)
        pipe2_Path = os.path.join(xl_results_dir, "faceid_0.7--plus_0.2", image_name)
        pipe3_Path = os.path.join(xl_results_dir, "faceid_0.7--plus_0.5", image_name)
        pipe4_Path = os.path.join(xl_results_dir, "faceid_0.7--facePlus_0.2", image_name)
        pipe5_Path = os.path.join(xl_results_dir, "faceid_0.7--facePlus_0.5", image_name)
        pipe6_Path = os.path.join(xl_results_dir, "faceid_0.7--control_0.2", image_name)
        pipe7_Path = os.path.join(xl_results_dir, "faceid_0.7--control_0.5", image_name)
        pipe8_Path = os.path.join(xl_results_dir, "faceid_0.7--control_0.2--facePlus_0.2", image_name)
        pipe9_Path = os.path.join(xl_results_dir, "faceid_0.7--control_0.2--plus_0.2", image_name)
        pipe10_Path = os.path.join(xl_results_dir, "faceid_0.7--control_0.5--facePlus_0.2", image_name)
        pipe11_Path = os.path.join(xl_results_dir, "faceid_0.7--control_0.5--plus_0.2", image_name)
        pipe12_Path = os.path.join(xl_results_dir, "faceid_0.7--plus_0.2--facePlus_0.2", image_name)
        pipe13_Path = os.path.join(xl_results_dir, "faceid_0.7--plus_0.2--facePlus_0.2--controlnet_0.2", image_name)
        
        if not (os.path.exists(pipe0_Path) and os.path.exists(pipe1_Path) and os.path.exists(pipe2_Path) and \
                os.path.exists(pipe3_Path) and os.path.exists(pipe4_Path) and os.path.exists(pipe5_Path) and \
                os.path.exists(pipe6_Path) and os.path.exists(pipe7_Path) and os.path.exists(pipe8_Path) and \
                os.path.exists(pipe9_Path) and os.path.exists(pipe10_Path) and os.path.exists(pipe11_Path) and \
                os.path.exists(pipe12_Path) and os.path.exists(pipe13_Path)):
            continue
        
        # print(f"image0 ==> {ori_path}")
        # print(f"image1 ==> {sd15_plusv2_image_path}")
        # print(f"image2 ==> {pipe0_Path}")
        # print(f"save_path ==> {save_path}")
        # exit(0)

        ori_image = np.array(resize_and_fill(ori_path, (target_size, target_size)).convert("RGB"))
        cv2.putText(ori_image, 'ori', (100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,255,0), thickness=3)
        sd15_plusv2_result = np.array(Image.open(sd15_plusv2_image_path).convert("RGB"))
        cv2.putText(sd15_plusv2_result, 'sd15_faceid_plus_v2', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.5, color=(255,0,0), thickness=2)
        sd15_plusv2_result = cv2.resize(sd15_plusv2_result, (target_size//2, target_size))
        
        pipe0_result = pic_frame(pipe0_Path, "pipe0--only_faceid_0.6", (target_size, target_size))
        pipe1_result = pic_frame(pipe1_Path, "pipe1--only_faceid_0.7", (target_size, target_size))
        pipe2_result = pic_frame(pipe2_Path, "pipe2--faceid_0.7--plus_0.2", (target_size, target_size))
        pipe3_result = pic_frame(pipe3_Path, "pipe3--faceid_0.7--plus_0.5", (target_size, target_size))
        pipe4_result = pic_frame(pipe4_Path, "pipe4--faceid_0.7--facePlus_0.2", (target_size, target_size))
        pipe5_result = pic_frame(pipe5_Path, "pipe5--faceid_0.7--facePlus_0.5", (target_size, target_size))
        pipe6_result = pic_frame(pipe6_Path, "pipe6--faceid_0.7--control_0.2", (target_size, target_size))
        pipe7_result = pic_frame(pipe7_Path, "pipe7--faceid_0.7--control_0.5", (target_size, target_size))
        pipe8_result = pic_frame(pipe8_Path, "pipe8--faceid_0.7--control_0.2--facePlus_0.2", (target_size, target_size))
        pipe9_result = pic_frame(pipe9_Path, "pipe9--faceid_0.7--control_0.2--plus_0.2", (target_size, target_size))
        pipe10_result = pic_frame(pipe10_Path, "pipe10--faceid_0.7--control_0.5--facePlus_0.2", (target_size, target_size))
        pipe11_result = pic_frame(pipe11_Path, "pipe11--faceid_0.7--control_0.5--plus_0.2", (target_size, target_size))
        pipe12_result = pic_frame(pipe12_Path, "pipe12--faceid_0.7--plus_0.2--facePlus_0.2", (target_size, target_size))
        pipe13_result = pic_frame(pipe13_Path, "pipe13--faceid_0.7--plus_0.2--facePlus_0.2--controlnet_0.2", (target_size, target_size))

        hconcat0 = cv2.hconcat([ori_image, sd15_plusv2_result, pipe0_result, pipe1_result])
        hconcat1 = cv2.hconcat([ori_image, sd15_plusv2_result, pipe2_result, pipe3_result])
        hconcat2 = cv2.hconcat([ori_image, sd15_plusv2_result, pipe4_result, pipe5_result])
        hconcat3 = cv2.hconcat([ori_image, sd15_plusv2_result, pipe6_result, pipe7_result])
        hconcat4 = cv2.hconcat([ori_image, sd15_plusv2_result, pipe8_result, pipe9_result])
        hconcat5 = cv2.hconcat([ori_image, sd15_plusv2_result, pipe10_result, pipe11_result])
        hconcat6 = cv2.hconcat([ori_image, sd15_plusv2_result, pipe12_result, pipe13_result])
        case = cv2.vconcat([hconcat0, hconcat1, hconcat2, hconcat3, \
                              hconcat4, hconcat5, hconcat6])
        # result = cv2.vconcat([result, case]) if result is not None else case
        save_path_ = save_path+f"_{os.path.basename(sd15_plusv2_image_path)}"
        Image.fromarray(case).save(save_path_)
        print(f"result has saving in {save_path_}")


if __name__ == '__main__':
    main()