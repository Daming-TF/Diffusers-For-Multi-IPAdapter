MODELDIR="/mnt/nfs/file_server/public/mingjiahui" 
export CUDA_VISIBLE_DEVICES=1,6
# \--config_file=default_config.yaml   # --num_processes 4 --multi_gpu --mixed_precision "fp16"
accelerate launch --config_file=./default_config_multi_gpu.yaml --main_process_port 29435  tutorial_train_faceid_costom_portrait.py \
    --pretrained_model_name_or_path="/mnt/nfs/file_server/public/mingjiahui/models/runwayml--stable-diffusion-v1-5" \
    --data_json_file="/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_2_crop.json" \
    --resolution=512 \
    --train_batch_size=32 \
    --dataloader_num_workers=2 \
    --learning_rate=1e-04 \
    --output_dir="$MODELDIR/experiments/faceid/finetune/20140126-sd15-crop-filter_reso448-V1_2" \
    --num_train_epochs=40 \
    --enable_xformers_memory_efficient_attention \
    --pretrained_ip_adapter_path="$MODELDIR/models/h94--IP-Adapter/h94--IP-Adapter/models/ip-adapter-faceid-portrait_sd15.bin" \
    --save_steps=2000 \
    --num_tokens=16
