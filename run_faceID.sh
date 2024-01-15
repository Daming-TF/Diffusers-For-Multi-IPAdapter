MODELDIR="/mnt/nfs/file_server/public/mingjiahui" 
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# \--config_file=default_config.yaml   # --num_processes 4 --multi_gpu --mixed_precision "fp16"
accelerate launch --config_file=default_config.yaml --main_process_port 29435  run_faceID.py \
    --pretrained_model_name_or_path="/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/" \
    --data_json_file="/mnt/nfs/file_server/public/mingjiahui/data/coyo700m/_tmp/train_coyo.json" \
    --resolution=1024 \
    --train_batch_size=4 \
    --dataloader_num_workers=0 \
    --learning_rate=1e-04 \
    --output_dir="$MODELDIR/experiments/faceid/finetune/faceid/0114" \
    --num_train_epochs=1 \
    --max_train_steps=40000 \
    --save_steps=100 \
    --num_token=4 \
    --enable_xformers_memory_efficient_attention \
    --pretrained_ip_adapter_path="$MODELDIR/models/h94--IP-Adapter/h94--IP-Adapter/sdxl_models/ip-adapter-faceid_sdxl.bin" \
    --checkpointing_steps=1000