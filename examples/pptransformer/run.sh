# Run Train
python main.py \
    enable_cinn=True


# Run Test, checkpoint need to be filled like
# ./output/Transolver/20250625_071023/model_150
python main.py \
    --config-name=transolver_shapenetcar.yaml \
    mode=test \
    checkpoint=./output/Transolver/your_checkpoint
