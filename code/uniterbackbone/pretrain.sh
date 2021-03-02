export PYTHONPATH=/data7/MEmoBert

CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python pretrain.py \
        --config config/pretrain-movies-v1-base-2gpu_rawimg_2optim_v1v2_res.json