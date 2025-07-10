# Run Train
python main_shapenetcar.py \
    --config-name=transolver_shapenetcar.yaml \
    data_module.data_dir="./data/preprocessed_data"

# Run Test, checkpoint need to be filled like
# ./output/Transolver/20250625_071023/model_150
python main_shapenetcar.py \
    --config-name=transolver_shapenetcar.yaml \
    data_module.data_dir="./data/preprocessed_data" \
    mode=test \
    checkpoint=./output/Transolver/checkpoint
