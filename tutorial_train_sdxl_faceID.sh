# export CUDA_VISIBLE_DEVICES=0,
accelerate launch --config_file=default_config.yaml ./tutorial_train_sdxl_faceID.py \
    --pretrained_model_name_or_path="/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/" \
    --data_json_file="/mnt/nfs/file_server/public/mingjiahui/data/coyo700m/_tmp/train_coyo.json" \
    --resolution=1024 \
    --train_batch_size=1 \
    --dataloader_num_workers=0 \
    --learning_rate=1e-05 \
    --weight_decay=0.01 \
    --output_dir="/mnt/nfs/file_server/public/mingjiahui/models/mjh_ipadater/sdxl_faceid/token_16/20240103" \
    --num_train_epochs=200000 \
    --save_steps=10000 \
    --only_load_adapter=True \
    --num_token=16 \
    # --print_freq=50 \
    # --use_wandb=True \
    # /mnt/nfs/file_server/public/mingjiahui/data/coyo700m/_tmp/train_coyo.json
    # --pretrained_ip_adapter_path="/mnt/nfs/file_server/public/mingjiahui/models/h94--IP-Adapter/h94--IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin" \
