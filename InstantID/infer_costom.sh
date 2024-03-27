export CUDA_VISIBLE_DEVICES=0
SOURCE_DIR="/mnt/nfs/file_server/public/mingjiahui/experiments/faceid/test_data/all_test_data/"
python instantid/infer_costom.py \
    --landmark_input=$SOURCE_DIR/aoteman.jpg \
    --faceid_input=$SOURCE_DIR/guonan.jpg 