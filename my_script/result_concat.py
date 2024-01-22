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
        # new_width = new_size[0]
        new_width = min(new_size)
        new_height = new_width*old_height/old_width
    else:
        # new_height = new_size[1]
        new_height = min(new_size)
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
    # ['3dpixel2sd', 'papercutoutsd', 'colorfulrhythmsd', 'joyfulcartoonsd']
    lora_id = "joyfulcartoonsd"
    test_data_id = "all_test_data"
    ori_input = fr'./data/{test_data_id}'

    xl_results_dir = fr"/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/untreated_output/{test_data_id}/{lora_id}"
    save_path = fr"/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/treated_result/compare_diff_i2i_t2i/{test_data_id}/{lora_id}"
    # save_path = "/home/mingjiahui/projects/IpAdapter/IP-Adapter/data/other/debug.jpg"
    os.makedirs(save_path, exist_ok=True)

    # 2. prepare data
    ori_results = [os.path.join(ori_input, name) for name in os.listdir(ori_input)]

    # 4. processing
    result = None
    for ori_path in tqdm(ori_results):
        image_name = os.path.basename(ori_path)

        t2i_pipe1_Path = os.path.join(xl_results_dir, "only_faceid_0.7", image_name)
        t2i_pipe2_Path = os.path.join(xl_results_dir, "only_faceid_plusv2_0.8", image_name)
        t2i_pipe3_Path = os.path.join(xl_results_dir, "faceid_plusV2_0.8--plus_0.3", image_name)
        t2i_pipe4_Path = os.path.join(xl_results_dir, "faceid_plusV2_0.8--controlnet0.2", image_name)
        t2i_pipe5_Path = os.path.join(xl_results_dir, "faceid_plusV2_0.8--plus_0.3--controlnet0.2", image_name)
        t2i_pipe6_Path = os.path.join(xl_results_dir, "faceid_plusV2_0.8--controlnet0.5", image_name)
        t2i_pipe7_Path = os.path.join(xl_results_dir, "faceid_plusV2_0.8--plus_0.3--controlnet0.5", image_name)
        
        i2i_pipe0_Path = os.path.join(xl_results_dir, "img2img--only_faceid_plusv2_0.8--strength_0.6", image_name)
        i2i_pipe1_Path = os.path.join(xl_results_dir, "img2img--only_faceid_plusv2_0.8--strength_1.0", image_name)
        i2i_pipe2_Path = os.path.join(xl_results_dir, "img2img--faceid_plusV2_0.8--plus_0.3--strength_0.6", image_name)
        i2i_pipe3_Path = os.path.join(xl_results_dir, "img2img--faceid_plusV2_0.8--plus_0.3--strength_1.0", image_name)
        i2i_pipe4_Path = os.path.join(xl_results_dir, "img2img--faceid_plusV2_0.8--controlnet0.2--strength_0.6", image_name)
        i2i_pipe5_Path = os.path.join(xl_results_dir, "img2img--faceid_plusV2_0.8--controlnet0.2--strength_1.0", image_name)
        i2i_pipe6_Path = os.path.join(xl_results_dir, "img2img--faceid_plusV2_0.8--plus_0.3--controlnet0.2--strength_0.6", image_name)
        i2i_pipe7_Path = os.path.join(xl_results_dir, "img2img--faceid_plusV2_0.8--plus_0.3--controlnet0.2--strength_1.0", image_name)
        i2i_pipe8_Path = os.path.join(xl_results_dir, "img2img--faceid_plusV2_0.8--controlnet0.5--strength_0.6", image_name)
        i2i_pipe9_Path = os.path.join(xl_results_dir, "img2img--faceid_plusV2_0.8--controlnet0.5--strength_1.0", image_name)
        i2i_pipe10_Path = os.path.join(xl_results_dir, "img2img--faceid_plusV2_0.8--plus_0.3--controlnet0.5--strength_0.6", image_name)
        i2i_pipe11_Path = os.path.join(xl_results_dir, "img2img--faceid_plusV2_0.8--plus_0.3--controlnet0.5--strength_1.0", image_name)
        
        if not (os.path.exists(t2i_pipe1_Path) and os.path.exists(t2i_pipe2_Path) and os.path.exists(t2i_pipe3_Path) and \
                os.path.exists(t2i_pipe4_Path) and os.path.exists(t2i_pipe5_Path) and os.path.exists(t2i_pipe6_Path) and \
                os.path.exists(t2i_pipe7_Path) and os.path.exists(i2i_pipe0_Path) and os.path.exists(i2i_pipe1_Path) and \
                os.path.exists(i2i_pipe2_Path) and os.path.exists(i2i_pipe3_Path) and os.path.exists(i2i_pipe4_Path) and \
                os.path.exists(i2i_pipe5_Path) and os.path.exists(i2i_pipe6_Path) and os.path.exists(i2i_pipe7_Path) and \
                os.path.exists(i2i_pipe8_Path) and os.path.exists(i2i_pipe9_Path) and os.path.exists(i2i_pipe10_Path) and \
                os.path.exists(i2i_pipe11_Path)):
            continue
        
        # print(f"image0 ==> {ori_path}")
        # print(f"image1 ==> {sd15_plusv2_image_path}")
        # print(f"image2 ==> {pipe0_Path}")
        # print(f"save_path ==> {save_path}")
        # exit(0)

        target_size_tuple = (target_size*2, target_size)
        ori_image = np.array(resize_and_fill(ori_path, target_size_tuple).convert("RGB"))
        cv2.putText(ori_image, 'ori', (100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0,255,0), thickness=3)
        
        t2i_pipe1_result = pic_frame(t2i_pipe1_Path, "t2ipipe1--only_faceid_0.7", target_size_tuple)
        t2i_pipe2_result = pic_frame(t2i_pipe2_Path, "t2ipipe2--only_faceid_plusv2_0.8", target_size_tuple)
        t2i_pipe3_result = pic_frame(t2i_pipe3_Path, "t2ipipe3--faceid_plusV2_0.8--plus_0.3", target_size_tuple)
        t2i_pipe4_result = pic_frame(t2i_pipe4_Path, "t2ipipe4--faceid_plusV2_0.8--controlnet0.2", target_size_tuple)
        t2i_pipe5_result = pic_frame(t2i_pipe5_Path, "t2ipipe5--faceid_plusV2_0.8--plus_0.3--controlnet0.2", target_size_tuple)
        t2i_pipe6_result = pic_frame(t2i_pipe6_Path, "t2ipipe6--faceid_plusV2_0.8--controlnet0.5", target_size_tuple)
        t2i_pipe7_result = pic_frame(t2i_pipe7_Path, "t2ipipe7--faceid_plusV2_0.8--plus_0.3--controlnet0.5", target_size_tuple)
        
        i2i_pipe0_result = pic_frame(i2i_pipe0_Path, "i2ipipe0--only_faceid_plusv2_0.8--strength_0.6", target_size_tuple)
        i2i_pipe1_result = pic_frame(i2i_pipe1_Path, "i2ipipe1--only_faceid_plusv2_0.8--strength_1.0", target_size_tuple)
        i2i_pipe2_result = pic_frame(i2i_pipe2_Path, "i2ipipe2--faceid_plusV2_0.8--plus_0.3--strength_0.6", target_size_tuple)
        i2i_pipe3_result = pic_frame(i2i_pipe3_Path, "i2ipipe3--faceid_plusV2_0.8--plus_0.3--strength_1.0", target_size_tuple)
        i2i_pipe4_result = pic_frame(i2i_pipe4_Path, "i2ipipe4--faceid_plusV2_0.8--controlnet0.2--strength_0.6", target_size_tuple)
        i2i_pipe5_result = pic_frame(i2i_pipe5_Path, "i2ipipe5--faceid_plusV2_0.8--controlnet0.2--strength_1.0", target_size_tuple)
        i2i_pipe6_result = pic_frame(i2i_pipe6_Path, "i2ipipe6--faceid_plusV2_0.8--plus_0.3--controlnet0.2--strength_0.6", target_size_tuple)
        i2i_pipe7_result = pic_frame(i2i_pipe7_Path, "i2ipipe7--faceid_plusV2_0.8--plus_0.3--controlnet0.2--strength_1.0", target_size_tuple)
        i2i_pipe8_result = pic_frame(i2i_pipe8_Path, "i2ipipe8--faceid_plusV2_0.8--controlnet0.5--strength_0.6", target_size_tuple)
        i2i_pipe9_result = pic_frame(i2i_pipe9_Path, "i2ipipe9--faceid_plusV2_0.8--controlnet0.5--strength_1.0", target_size_tuple)
        i2i_pipe10_result = pic_frame(i2i_pipe10_Path, "i2ipipe10--faceid_plusV2_0.8--plus_0.3--controlnet0.5--strength_0.6", target_size_tuple)
        i2i_pipe11_result = pic_frame(i2i_pipe11_Path, "i2ipipe11--faceid_plusV2_0.8--plus_0.3--controlnet0.5--strength_1.0", target_size_tuple)

        hconcat0 = cv2.hconcat([ori_image, t2i_pipe1_result])
        hconcat1 = cv2.hconcat([t2i_pipe2_result, t2i_pipe3_result])
        hconcat2 = cv2.hconcat([t2i_pipe4_result, t2i_pipe5_result])
        hconcat3 = cv2.hconcat([t2i_pipe6_result, t2i_pipe7_result])
        hconcat4 = cv2.hconcat([i2i_pipe0_result, i2i_pipe1_result])
        hconcat5 = cv2.hconcat([i2i_pipe2_result, i2i_pipe3_result])
        hconcat6 = cv2.hconcat([i2i_pipe4_result, i2i_pipe5_result])
        hconcat7 = cv2.hconcat([i2i_pipe6_result, i2i_pipe7_result])
        hconcat8 = cv2.hconcat([i2i_pipe8_result, i2i_pipe9_result])
        hconcat9 = cv2.hconcat([i2i_pipe10_result, i2i_pipe11_result])
        case = cv2.vconcat([hconcat0, hconcat1, hconcat2, hconcat3, \
                              hconcat4, hconcat5, hconcat6, hconcat7, hconcat8, hconcat9])
        # result = cv2.vconcat([result, case]) if result is not None else case
        save_path_ = os.path.join(save_path, f"{os.path.basename(ori_path)}")
        Image.fromarray(case).save(save_path_)
        print(f"result has saving in {save_path_}")


if __name__ == '__main__':
    main()