export PYTHONPATH=/data4/MEmoBert
gpu_id=$1
frozens=0
dropout=0.1

# for iemocap, trnsize = 3500/64 * 6 = 500 if bs=32, then use 800
corpus_name='msp'
corpus_name_big='MSP'
for cvNo in `seq 7 12`;
do
CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
        --cvNo ${cvNo} --model_config config/uniter-base-emoword_multitask.json \
        --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
        --checkpoint /data4/emobert/exp/pretrain/nomask_movies_v1v2_uniter_4tasks_lr5e5_bs1024_faceth0.5_mlt-melm5/ckpt/model_step_5000.pt \
        --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix run2 \
        --learning_rate 2e-5 --lr_sched_type 'linear' --conf_th 0.0 --max_bb 36 \
        --train_batch_size 64 --train_batch_size 64 --num_train_steps 800 \
        --output_dir /data4/emobert/exp/evaluation/${corpus_name_big}/finetune/baseon-movies_v1v2_uniter_4tasks_mltmelm5_step5k-lr2e5_bs64_max36
done

# for iemocap, trnsize = 5000/64 * 6 = 600 if bs=32, then use 1000
# corpus_name='iemocap'
# corpus_name_big='IEMOCAP'
# for cvNo in `seq 9 10`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#         --cvNo ${cvNo} --model_config config/uniter-base-emoword_multitask.json \
#         --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#         --checkpoint /data4/emobert/exp/pretrain/nomask_movies_v1v2_uniter_2tasks_lr5e5_bs1024_faceth0.5_mlt-melm5/ckpt/model_step_5000.pt \
#         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix run2 \
#         --learning_rate 2e-5 --lr_sched_type 'linear' --conf_th 0.0 --max_bb 64 \
#         --train_batch_size 128 --train_batch_size 128 --num_train_steps 600 \
#         --output_dir /data4/emobert/exp/evaluation/${corpus_name_big}/finetune/baseon-movies_v1v2_uniter_2tasks_mltmelm5_step5k-lr2e5_bs128_max64
# done


# 为了少修改代码，这里将meld放在cvNo=1下面
# corpus_name='meld'
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#         --cvNo 1 --model_config config/uniter-base-emoword_multitask.json \
#         --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#         --checkpoint /data4/emobert/exp/pretrain/nomask_movies_v1v2_uniter_3tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_5000.pt \
#         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
#         --learning_rate 2e-5 --lr_sched_type 'linear' --conf_th 0.5 \
#         --train_batch_size 128 --train_batch_size 128   --num_train_steps 1000 \
#         --output_dir /data4/emobert/exp/evaluation/MELD/finetune/baseon-movies_v1v2_uniter_3tasks_step5k-lr2e5_bs128_th0.5