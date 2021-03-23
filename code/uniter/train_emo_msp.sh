export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
frozens=0
dropout=0.1

# for msp, trnsize = 3500/32 * 10 = 1200
# corpus_name='msp'
# corpus_name_big='MSP'
# for cvNo in `seq 1 12`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#         --cvNo ${cvNo} --model_config config/uniter-base-emoword_nomultitask.json \
#         --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2_uniter_3tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_5000.pt \
#         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
#         --learning_rate 2e-5 --lr_sched_type 'linear' --conf_th 0.0 --max_bb 36 \
#         --train_batch_size 32 --train_batch_size 32 --num_train_steps 1200 \
#         --output_dir /data7/emobert/exp/evaluation/${corpus_name_big}/finetune/baseon-movies_v1v2_uniter_3tasks-lr2e5_bs32_max36_train1200
# done

# corpus_name='msp'
# corpus_name_big='MSP'
# for cvNo in `seq 7 12`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#         --cvNo ${cvNo} --model_config config/uniter-base-emoword_nomultitask.json \
#         --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2_uniter_2tasks_lr5e5_bs1024_faceth0.5_mlt-melm5/ckpt/model_step_5000.pt \
#         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
#         --learning_rate 2e-5 --lr_sched_type 'linear' --conf_th 0.0 --max_bb 36 \
#         --train_batch_size 32 --train_batch_size 32 --num_train_steps 1200 \
#         --output_dir /data7/emobert/exp/evaluation/${corpus_name_big}/finetune/baseon-movies_v1v2_uniter_2tasks_mltmelm5-lr2e5_bs32_max36_train1200
# done

# for taskpretrain, trnsize = 3500/32 * 10 = 1200 if bs=32(better)
corpus_name='msp'
corpus_name_big='MSP'
for cvNo in `seq 1 12`;
do
CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
        --cvNo ${cvNo} --model_config config/uniter-base-emoword_nomultitask.json \
        --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
        --checkpoint /data7/emobert/exp/task_pretrain/msp_basedon-nomask_movies_v1v2_uniter_4tasks_faceth0.5_5k-4tasks_maxbb36_faceth0.0/${cvNo}/ckpt/model_step_400.pt \
        --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
        --learning_rate 2e-5 --lr_sched_type 'linear' --conf_th 0.0 --max_bb 36 \
        --train_batch_size 32 --train_batch_size 32 --num_train_steps 1200 --warmup_steps 0 --valid_steps 200 \
        --output_dir /data7/emobert/exp/evaluation/${corpus_name_big}/finetune/baseon-taskpretain_movies_v1v2_uniter_4task_train400-lr2e5_bs32
done

# corpus_name='msp'
# corpus_name_big='MSP'
# for cvNo in `seq 1 12`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#         --cvNo ${cvNo} --model_config config/uniter-base-emoword_nomultitask.json \
#         --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#         --checkpoint /data7/emobert/exp/task_pretrain/msp_basedon-nomask_movies_v1v2_uniter_4tasks_faceth0.5_5k-4tasks_maxbb36_faceth0.0_mltmelm5/${cvNo}/ckpt/model_step_400.pt \
#         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
#         --learning_rate 2e-5 --lr_sched_type 'linear' --conf_th 0.0 --max_bb 36 \
#         --train_batch_size 32 --train_batch_size 32 --num_train_steps 1200 --warmup_steps 0 --valid_steps 200 \
#         --output_dir /data7/emobert/exp/evaluation/${corpus_name_big}/finetune/baseon-taskpretain_movies_v1v2_uniter_4tasks_mltmelm5_train400-lr2e5_bs32
# done