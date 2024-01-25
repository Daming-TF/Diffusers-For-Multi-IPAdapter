MODELDIR="/mnt/nfs/file_server/public/mingjiahui" 
export CUDA_VISIBLE_DEVICES=1,2,3,4
# \--config_file=default_config.yaml   # --num_processes 4 --multi_gpu --mixed_precision "fp16"
accelerate launch --config_file=default_config.yaml --main_process_port 29435  tutorial_train_faceid_costom_portrait.py \
    --pretrained_model_name_or_path="/mnt/nfs/file_server/public/mingjiahui/models/runwayml--stable-diffusion-v1-5" \
    --data_json_file="/mnt/nfs/file_server/public/mingjiahui/data/coyo700m/_tmp/train-coyo.json" \
    --resolution=512 \
    --train_batch_size=8 \
    --dataloader_num_workers=0 \
    --learning_rate=1e-04 \
    --output_dir="$MODELDIR/experiments/faceid/finetune/20140122-sd15" \
    --num_train_epochs=200000 \
    --enable_xformers_memory_efficient_attention \
    --pretrained_ip_adapter_path="$MODELDIR/models/h94--IP-Adapter/h94--IP-Adapter/models/ip-adapter-faceid-portrait_sd15.bin" \
    --save_steps=2000 \
