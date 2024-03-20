MODELDIR="/mnt/nfs/file_server/public/mingjiahui" 
export CUDA_VISIBLE_DEVICES=1,
# \--config_file=default_config.yaml   # --num_processes 4 --multi_gpu --mixed_precision "fp16"
# default_config_single_gpu.yaml
# default_config_multi_gpu.yaml
# /mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_crop.json
accelerate launch --config_file=./default_config_single_gpu.yaml --main_process_port 29438  ./tutorial_train_faceid_costom_styleGAN.py \
    --pretrained_model_name_or_path="/mnt/nfs/file_server/public/mingjiahui/models/runwayml--stable-diffusion-v1-5" \
    --data_json_file="/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_crop_styleGAN.json" \
    --resolution=512 \
    --train_batch_size=32 \
    --dataloader_num_workers=4 \
    --learning_rate=1e-04 \
    --output_dir="$MODELDIR/experiments/faceid/finetune/base-portrait-styleGAN/20240303-sd15-crop--V1-wo_xformer-scratch" \
    --num_train_epochs=40 \
    --save_steps=1 \
    --deepface_run_step=1 \
    --image_encoder='buffalo_l' \
    --pretrained_ip_adapter_path="$MODELDIR/models/styleGAN/wplus_adapter.bin" \
    --sr
    # --enable_xformers_memory_efficient_attention \
 