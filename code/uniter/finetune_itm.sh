CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python train_itm_hard_negatives.py \
        --config config/train-itm-coco-base-8gpu-hn.json
