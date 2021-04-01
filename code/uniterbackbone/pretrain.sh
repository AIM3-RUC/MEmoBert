export PYTHONPATH=/data7/MEmoBert

# 1. conv3dresnet-pretrained-oneOptim-th0.5 --running
# 2. conv3dresnet-pretrained-twoOptim-th0.5
# 3. conv3dresnet-fromscratch-oneOptim-th0.5
# 4. conv3dresnet-fromscratch-twoOptim-th0.5

# batchsize=180000/40/ *20 = 
CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
        --config config/pretrain-movies-v1v2v3-base-2gpu_rawimg_2optim_mlmitm.json \
        --model_config config/uniter-base-backbone_3dresnet_pretrained.json \
        --train_batch_size 30 --val_batch_size 30 --gradient_accumulation_steps 2 \
        --learning_rate 5e-5 --lr_sched_type 'linear' \
        --max_txt_len 30 --IMG_DIM 112 \
        --conf_th 0.5 --max_bb 36 \
        --use_backbone_optim false \
        --num_train_steps 8000 --warmup_steps 800 --valid_steps 800 \
        --output_dir /data7/emobert/exp/pretrain/resnet3d_nomask_movies_v1v2v3_uniter_mlmitm_lr5e5-backbone_pretrained_optimFalse-bs512_faceth0.5 

# CUDA_VISIBLE_DEVICES=2,3 horovodrun -np 2 python pretrain.py \
# CUDA_VISIBLE_DEVICES=5,7 horovodrun -np 2 python pretrain.py \