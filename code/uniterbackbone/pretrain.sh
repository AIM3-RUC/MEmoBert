export PYTHONPATH=/data7/MEmoBert

CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
        --config config/pretrain-movies-v1-base-2gpu_rawimg_2optim.json