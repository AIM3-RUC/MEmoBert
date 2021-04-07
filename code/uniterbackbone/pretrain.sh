export PYTHONPATH=/data7/MEmoBert

# 1. conv3dresnet-pretrained-oneOptim-th0.5 --running
# 2. conv3dresnet-pretrained-twoOptim-th0.5 --done
# 3. conv3dresnet-fromscratch-oneOptim-th0.5 --running
# 4. conv3dresnet-fromscratch-twoOptim-th0.5 --done

# batchsize=180000/(30*2*8) *20 = 7500
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --config config/pretrain-movies-v1v2v3-base-2gpu_rawimg_2optim_mlmitm.json \
#         --model_config config/uniter-base-backbone_3dresnet_scratch.json \
#         --train_batch_size 30 --val_batch_size 30 --gradient_accumulation_steps 8 \
#         --learning_rate 1e-4 --lr_sched_type 'linear' \
#         --max_txt_len 30 --IMG_DIM 112 --image_data_augmentation \
#         --conf_th 0.5 --max_bb 36 \
#         --num_train_steps 10000 --warmup_steps 1000 --valid_steps 1000 \
#         --output_dir /data7/emobert/exp/pretrain/resnet3d_nomask_movies_v1v2v3_uniter_mlmitm_lr1e4-backbone_pretrained_optimFalse-bs480_faceth0.5

CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
        --cvNo 0 --config config/pretrain-movies-v1v2v3-base-2gpu_rawimg_2optim_mlmitm.json \
        --model_config config/uniter-base-backbone_3dresnet.json \
        --checkpoint /data7/MEmoBert/emobert/exp/pretrain/resnet3d_nomask_movies_v1v2v3_uniter_mlmitm_lr5e5-backbone_scratch_optimFalse-bs480_faceth0.5/ckpt/model_step_10000.pt \
        --is_reinit_lr \
        --train_batch_size 30 --val_batch_size 30 --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 --lr_sched_type 'linear' \
        --max_txt_len 30 --IMG_DIM 112 --image_data_augmentation \
        --conf_th 0.5 --max_bb 36 \
        --num_train_steps 10000 --warmup_steps 1000 --valid_steps 1000 \
        --output_dir /data7/emobert/exp/pretrain/resnet3d_nomask_movies_v1v2v3_uniter_mlmitm_lr5e5-backbone_scratch_optimFalse-bs480_faceth0.5_continue1w

# CUDA_VISIBLE_DEVICES=3,7 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_rawimg_2optim_mlmitm.json \
#         --model_config config/uniter-base-backbone_3dresnet_scratch.json \
#         --train_batch_size 30 --val_batch_size 30 --gradient_accumulation_steps 8 \
#         --learning_rate 5e-5 --lr_sched_type 'linear' \
#         --max_txt_len 30 --IMG_DIM 112 --image_data_augmentation \
#         --conf_th 0.5 --max_bb 36 \
#         --num_train_steps 8000 --warmup_steps 1000 --valid_steps 1000 \
#         --output_dir /data7/emobert/exp/pretrain/resnet3d_nomask_movies_v1v2v3_uniter_mlmitm_lr5e5-backbone_scratch_optimFalse-bs480_faceth0.5 

# CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python pretrain.py \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_rawimg_2optim_mlmitm.json \
#         --model_config config/uniter-base-backbone_3dresnet_scratch.json \
#         --train_batch_size 30 --val_batch_size 30 --gradient_accumulation_steps 8 \
#         --learning_rate 5e-5 --lr_sched_type 'linear' \
#         --max_txt_len 30 --IMG_DIM 112 --image_data_augmentation true \
#         --conf_th 0.5 --max_bb 36 \
#         --use_backbone_optim true  --backbone_learning_rate 5e-4 \
#         --num_train_steps 8000 --warmup_steps 1000 --valid_steps 1000 \
#         --output_dir /data7/emobert/exp/pretrain/resnet3d_nomask_movies_v1v2v3_uniter_mlmitm_lr5e5-backbone_scratch_optimTrue_lr5e4-bs480_faceth0.5 