export PYTHONPATH=/data4/MEmoBert
gpu_id=$1
frozens=$2
dropout=$3

# corpus_name='iemocap'
# for i in `seq 1 10`;
# do
#     CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#             --cvNo ${i} --config config/train-emo-${corpus_name}-openface-base-4gpu.json \
#             --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_4tasks_faceth0.1_melm0.5/ckpt/model_step_10000.pt \
#             --output_dir /data7/emobert/exp/finetune/${corpus_name}_openface_baseon_nomask_movies_v1_uniter_4tasks_melm0.5_10k_fix5e5 \
#             --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
#             --learning_rate 5e-5 --train_batch_size 80
# done



# 为了少修改代码，这里将meld放在cvNo=1下面
corpus_name='meld'
CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
        --cvNo 1 --model_config config/uniter-base-emoword_multitask.json \
        --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
        --checkpoint /data4/emobert/exp/pretrain/nomask_movies_v1v2_uniter_3tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_5000.pt \
        --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
        --learning_rate 2e-5 --lr_sched_type 'linear' --conf_th 0.5 \
        --train_batch_size 128 --train_batch_size 128   --num_train_steps 1000 \
        --output_dir /data4/emobert/exp/evaluation/MELD/finetune/baseon-movies_v1v2_uniter_3tasks_step5k-lr2e5_bs128_th0.5