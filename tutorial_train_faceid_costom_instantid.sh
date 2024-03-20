MODELDIR="/mnt/nfs/file_server/public/mingjiahui" 
export CUDA_VISIBLE_DEVICES=0,1
# ./default_config_multi_gpu.yaml
# ./default_config_single_gpu.yaml
accelerate launch --config_file=./default_config_multi_gpu.yaml --main_process_port 29438 tutorial_train_faceid_costom_instantid.py \
    --pretrained_model_name_or_path="/mnt/nfs/file_server/public/mingjiahui/models/runwayml--stable-diffusion-v1-5" \
    --data_json_file="/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/train_json/traindata_V1_with_all_face_info.json" \
    --resolution=512 \
    --train_batch_size=16 \
    --dataloader_num_workers=4 \
    --learning_rate=1e-5 \
    --output_dir="$MODELDIR/experiments/faceid/finetune/instantid-portrait-condition-txt_image/20240320-sd15--V1-wo_xformer-pretrain_from_step24000" \
    --num_train_epochs=40 \
    --save_steps=1500 \
    --num_tokens=16 \
    --deepface_run_step=1500 \
    --factor=2 \
    --gradient_accumulation_steps=1 \
    --control_condition=txt_image \
    --controlnet_model_name_or_path="$MODELDIR/experiments/faceid/finetune/instantid-portrait-condition-txt_image/20240320-sd15--V1-wo_xformer-scratch/checkpoint-24000" \
    --pretrained_ip_adapter_path="$MODELDIR/experiments/faceid/finetune/instantid-portrait-condition-txt_image/20240320-sd15--V1-wo_xformer-scratch/checkpoint-24000/sd15_instantid.bin" \
    # --enable_xformers_memory_efficient_attention \
