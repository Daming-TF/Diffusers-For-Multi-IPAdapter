export CUDA_VISIBLE_DEVICES=0
SOURCE_DIR="/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/all_test_data/"
python ./my_script/faceid_experiment/instantid_script_separate_controlnet_ip.py \
    --landmark_input=$SOURCE_DIR/aoteman.jpg \