export PYTHONPATH=/data7/MEmoBert

CUDA_VISIBLE_DEVICES=3,4 horovodrun -np 2 python pretrain.py \
        --config config/pretrain-movies-v1v2-base-2gpu_rawimg_2optim_res.json