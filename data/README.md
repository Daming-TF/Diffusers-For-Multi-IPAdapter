# 数据处理逻辑
1. arcface_process.py      
    多进程多线程处理数据，计算faceid embeds并保存到指定路径

2. data_statistics.py      
    根据.npy文件统计指定数据集中单人脸多人脸在各个分辨率的分布，每个进程按照一下键名统计数据：“512,640,768,896,1024，surpass_1024, single, multi”,分辨率键名存放对应图片路径, 'single'和'multi'统计的是分辨率分布情况，如果设置'transfer_to_train'为True，会把指定reso的图片路径合并到list保存成训练格式的json

3. filter_bbox.py
    输入训练格式的json，根据bbox对图片crop操作并保存crop image，并记录crop_reso加上原本json信息一起保存成 ***_crop.json

4. train_json_concat.py
    输入多个数据的训练格式json文件并合并

5. data_statistics_for_train_json.py       
    ststis模式  输入训练的json文件，统计样本分辨率，每个进程各自生成结果文件存放在_tmp
    check模式   输入训练的json文件，随机采样n个样本你检查
    concat模式  选择需要的分辨率阈值，根据_tmp文件夹内容合并文件

6. get_106_2d_kps.py
    输入训练格式json文件，多进程处理得到对应的106*2的关键点坐标，并保存到.npy文件

ori_image处理顺序：1,2,4
crop_image处理顺序：1,2,3,4,5
image_kps处理顺序：1,2,4,6


# 其他逻辑
1. data_prepare_decompression_files.py     
    多进程解压

2. result_concat.py
    用于多个pipeline结果对比的合成工具
