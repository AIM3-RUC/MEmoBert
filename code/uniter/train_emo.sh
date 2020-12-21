export PYTHONPATH=/data7/MEmoBert
CUDA_VISIBLE_DEVICES=4 horovodrun -np 1 python train_emo.py \
        --config config/train-emo-iemocap-base-4gpu.json