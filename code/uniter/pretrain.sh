CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
        --config config/pretrain-movies-v1-base-8gpu.json