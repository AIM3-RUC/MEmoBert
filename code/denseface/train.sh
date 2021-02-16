export PYTHONPATH=/data7/MEmoBert
set -e
gpu=$1
python train.py --gpu_id $gpu --config_file config/conf_fer.py \
    --learning_rate 0.001 --drop_rate 0.0