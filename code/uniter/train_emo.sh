CUDA_VISIBLE_DEVICES=0,1,2,3 horovodrun -np 4 python train_emo.py \
        --config config/train-emo-iemocap-base-4gpu.json