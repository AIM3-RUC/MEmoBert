CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python pretrain.py \
        --config config/pretrain-movies-v1-base-8gpu.json