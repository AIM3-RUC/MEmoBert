# single model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 horovodrun -np 8 python pretrain.py \
        --config config/pretrain-indomain-base-8gpu.json
