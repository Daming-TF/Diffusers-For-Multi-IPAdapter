MODELDIR="/mnt/nfs/file_server/public/mingjiahui" 
export CUDA_VISIBLE_DEVICES=2
# ./default_config_multi_gpu.yaml
# ./default_config_single_gpu.yaml
accelerate launch --config_file=./default_config_single_gpu.yaml --main_process_port 29436 tutorial_train_faceid_costom_portrait_ti_token.py \
    --pretrained_model_name_or_path="/mnt/nfs/file_server/public/mingjiahui/models/runwayml--stable-diffusion-v1-5" \
    --image_encoder_path="$MODELDIR/models/h94--IP-Adapter/h94--IP-Adapter/models/image_encoder" \
    --data_json_file="/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_crop.json" \
    --resolution=512 \
    --train_batch_size=32 \
    --dataloader_num_workers=4 \
    --learning_rate=5e-5 \
    --output_dir="$MODELDIR/experiments/faceid/finetune/portrait-ti_token/20240208-sd15-crop--V1-wo_xformer-scratch" \
    --num_train_epochs=40 \
    --save_steps=2500 \
    --num_tokens=16 \
    --ti_num_tokens=4 \
    --pretrained_ip_adapter_path="$MODELDIR/experiments/faceid/finetune/base-portrait/20140205-sd15-crop--V1-wo_xformer-scratch_from_step190000/checkpoint-74000/sd15_faceid_portrait.bin" \
    --deepface_run_step=2500 \
    --report_to="wandb" \
    # --enable_xformers_memory_efficient_attention \
