export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
frozens=0
dropout=0.1
corpus_name='meld'

# for meld, directly pretrain, trnsize = 10000/32 * 10 = 3000
# 为了少修改代码，这里将meld放在cvNo=1下面
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#         --cls_num 7 \
#         --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
#         --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2_uniter_2tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_5000.pt \
#         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
#         --learning_rate 2e-5 --lr_sched_type 'linear' --conf_th 0.5 \
#         --train_batch_size 32 --train_batch_size 32 --num_train_steps 3000 --valid_steps 300  \
#         --output_dir /data7/emobert/exp/evaluation/MELD/finetune/baseon-movies_v1v2_uniter_4tasks-lr2e5_bs32_th0.5_train3000

# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#         --cls_num 7 \
#         --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
#         --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2_uniter_4tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_5000.pt \
#         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
#         --learning_rate 2e-5 --lr_sched_type 'linear' --conf_th 0.1 --max_bb 36 \
#         --train_batch_size 32 --train_batch_size 32 --num_train_steps 3000 --valid_steps 300  \
#         --output_dir /data7/emobert/exp/evaluation/MELD/finetune/baseon-movies_v1v2_uniter_4tasks-lr2e5_bs32_th0.1_train3000

# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#         --cls_num 7 \
#         --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
#         --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2_uniter_2tasks_lr5e5_bs1024_faceth0.5_mlt-melm5/ckpt/model_step_5000.pt \
#         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
#         --learning_rate 2e-5 --lr_sched_type 'linear' --conf_th 0.1 --max_bb 36 \
#         --train_batch_size 32 --train_batch_size 32 --num_train_steps 3000 --valid_steps 300  \
#         --output_dir /data7/emobert/exp/evaluation/MELD/finetune/baseon-movies_v1v2_uniter_2tasks_mltmelm5-lr2e5_bs32_th0.1_train3000

# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#         --cls_num 7 \
#         --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
#         --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2_uniter_2tasks_lr5e5_bs1024_faceth0.5_mlt-melm5/ckpt/model_step_5000.pt \
#         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
#         --learning_rate 2e-5 --lr_sched_type 'linear' --conf_th 0.5 --max_bb 36 \
#         --train_batch_size 32 --train_batch_size 32 --num_train_steps 3000 --valid_steps 300  \
#         --output_dir /data7/emobert/exp/evaluation/MELD/finetune/baseon-movies_v1v2_uniter_2tasks_mltmelm5-lr2e5_bs32_th0.5_train3000

# ## for task pretrained model, thface0.1
corpus_name='meld'
CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
        --cls_num 7 \
        --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
        --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
        --checkpoint /data7/emobert/exp/task_pretrain/meld_basedon-nomask_movies_v1v2_uniter_3tasks_faceth0.5_mltmelm5_5k-4tasks_maxbb36_faceth0.1_mltmelm5/1/ckpt/model_step_800.pt \
        --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
        --learning_rate 2e-5 --lr_sched_type 'linear' --conf_th 0.1 \
        --train_batch_size 32 --train_batch_size 32   --num_train_steps 3000 \
        --output_dir /data7/emobert/exp/evaluation/MELD/finetune/baseon-taskpretrain-movies_v1v2_uniter_3tasks_faceth0.1_mltmelm5_step600-lr2e5_bs32_faceth0.1

corpus_name='meld'
CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
        --cls_num 7 \
        --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
        --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
        --checkpoint /data7/emobert/exp/task_pretrain/meld_basedon-nomask_movies_v1v2_uniter_3tasks_faceth0.5_mltmelm5_5k-4tasks_maxbb36_faceth0.5_mltmelm5/1/ckpt/model_step_800.pt \
        --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
        --learning_rate 2e-5 --lr_sched_type 'linear' --conf_th 0.5 \
        --train_batch_size 32 --train_batch_size 32   --num_train_steps 3000 \
        --output_dir /data7/emobert/exp/evaluation/MELD/finetune/baseon-taskpretrain-movies_v1v2_uniter_3tasks_faceth0.5_mltmelm5_step800-lr2e5_bs32_faceth0.5