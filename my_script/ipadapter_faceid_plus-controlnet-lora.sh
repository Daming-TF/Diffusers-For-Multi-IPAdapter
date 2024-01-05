export CUDA_VISIBLE_DEVICES=$1,
python my_script/ipadapter_faceid_plus-controlnet-lora.py \
    --input ./data/conventional_testing/wangbaoqiang.jpg \
    --output ./data/conventional_testing_control_lora_output/joyful_cartoon \
    --prompt="closeup photo of a man wearing a white shirt in a garden" \
    --v2 \
    --faceid_lora_weight=0.6 \
    --lora="joyful_cartoon" \
    --lora_weight=1.0 \
    --control_mode=hed \
    --control_scale=$2 \
    
