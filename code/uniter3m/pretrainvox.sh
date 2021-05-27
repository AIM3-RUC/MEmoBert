export PYTHONPATH=/data7/MEmoBert

### for voxceleb2

## case1: visual + text running on gpu0, only voxceleb2
# CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-vox2-v1-base-2gpu_4tasks_emo_sentiword.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 60 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 4000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5

## case2: visual + text running on gpu0,  moviesv1v2v3 + voxceleb2
# CUDA_VISIBLE_DEVICES=3,4 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_4tasks_emo_sentiword.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 60 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 4000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs800_faceth0.5

## case3: visual + text + emocls running on gpu0, moviesv1v2v3 + voxceleb2 
# CUDA_VISIBLE_DEVICES=5,6 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_4tasks_emo_sentiword_emocls.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 60 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 4000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_emoclsSoft_lr5e5_bs800_faceth0.5


## case4: visual + text + emocls running on gpu0, moviesv1v2v3 + voxceleb2 + projs
CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
        --cvNo 0 --n_workers 4  --use_visual  \
        --config config/pretrain-movies-v1v2v3-vox2-v1-base-2gpu_4tasks_emo_sentiword_emocls.json \
        --model_config config/uniter-base-emoword_nomultitask_difftype_pojs_weaklabelSoft.json \
        --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
        --max_txt_len 60 \
        --IMG_DIM 342 --Speech_DIM 768 \
        --train_batch_size 200 --val_batch_size 200 \
        --num_train_steps 40000 --warmup_steps 4000 --valid_steps 4000 \
        --output_dir /data7/emobert/exp/pretrain/nomask_moviesv1v2v3_vox2_v1_uniter3m_visual_text_4tasks_emoclsSoft_avprojs_lr5e5_bs800_faceth0.5