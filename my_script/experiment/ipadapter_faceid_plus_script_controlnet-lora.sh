export CUDA_VISIBLE_DEVICES=$1,
python my_script/experiment/ipadapter_faceid_plus_script_controlnet-lora.py \
    --input ./data/all_test_data \
    --output ./data/all_test_data_output/lora_controlnet_script/$2 \
    --prompt="closeup photo of a persion wearing a white shirt in a garden" \
    --v2 \
    --faceid_lora_weight=1.0 \
    --lora=$2 \
    --lora_weights="0.4-1.0-0.2" \
    --control_mode=hed \
    --control_weights="0-0.5-0.1" \
    
