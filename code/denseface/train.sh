export PYTHONPATH=/data7/MEmoBert
set -e
gpu=$1

#### for train 
python train.py --gpu_id $gpu --config_file config/conf_fer.py \
    --learning_rate 0.001 --drop_rate 0.0

#### for evaluation
# python train.py --gpu_id $gpu --config_file config/conf_fer.py \
#     --learning_rate 0.001 --drop_rate 0.0 --is_test \
#     --restore_checkpoint /data7/emobert/exp/face_model/densenet100_adam0.001_0.0/ckpts/model_step_43.pt