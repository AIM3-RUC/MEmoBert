export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
frozens=0
dropout=0.1
corpus_name='iemocap'
corpus_name_big='IEMOCAP'

# for iemocap, directly pretrain no melm, trnsize = 5000/32 * 10 = 1600
# for max_bb in 36 64;
# do
#         for lr in 1e-5 2e-5 5e-5; 
#         do
#                 for cvNo in `seq 1 10`;
#                 do
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                         --cvNo ${cvNo} --model_config config/uniter-base-emoword_nomultitask.json \
#                         --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_6000.pt \
#                         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
#                         --learning_rate ${lr} --lr_sched_type 'linear' --conf_th 0.0 --max_bb ${max_bb} \
#                         --train_batch_size 32 --train_batch_size 32 --num_train_steps 1600 \
#                         --output_dir /data7/emobert/exp/evaluation/${corpus_name_big}/finetune/baseon-movies_v1v2v3_uniter_4tasks-lr${lr}_bs32_max${max_bb}_train1600
#                 done
#         done
# done


# for iemocap, directly pretrain with melm, trnsize = 5000/32 * 10 = 1600
for max_bb in 36 64;
do
   for lr in 1e-5 2e-5 5e-5; 
   do
        for cvNo in `seq 1 10`;
        do
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
                --cvNo ${cvNo} --model_config config/uniter-base-emoword_multitask.json \
                --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
                --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_lr5e5_bs1024_faceth0.5_mlt-melm5/ckpt/model_step_6000.pt \
                --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
                --learning_rate ${lr} --lr_sched_type 'linear' --conf_th 0.0 --max_bb ${max_bb} \
                --train_batch_size 32 --train_batch_size 32 --num_train_steps 1600 \
                --output_dir /data7/emobert/exp/evaluation/${corpus_name_big}/finetune/baseon-movies_v1v2v3_uniter_4tasks_mltmelm5-lr${lr}_bs32_max${max_bb}_train1600
        done
   done
done

# for iemocap task pretrain, trnsize = 5000/64 * 6 = 600 if bs=32, then use 1000
# corpus_name='iemocap'
# corpus_name_big='IEMOCAP'
# for cvNo in `seq 9 10`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#         --cvNo ${cvNo} --model_config config/uniter-base-emoword_multitask.json \
#         --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#         --checkpoint /data7/emobert/exp/task_pretrain/iemocap_basedon-nomask_movies_v1v2_uniter_4tasks_faceth0.5_mltmelm5_5k-4tasks_maxbb64_faceth0.0_mltmelm5/${cvNo}/ckpt/model_step_400.pt \
#         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
#         --learning_rate 2e-5 --lr_sched_type 'linear' --conf_th 0.0 --max_bb 64 \
#         --train_batch_size 128 --train_batch_size 128 --num_train_steps 1600 \
#         --output_dir /data7/emobert/exp/evaluation/${corpus_name_big}/finetune/baseon-taskpretain_movies_v1v2_uniter_4tasks_mltmelm5_train400-lr2e5_bs128_max64_train1600
# done
# --checkpoint /data7/emobert/exp/task_pretrain/iemocap_basedon-nomask_movies_v1v2_uniter_4tasks_faceth0.5_5k-4tasks_maxbb64_faceth0.0/${cvNo}/ckpt/model_step_400.pt \
# --output_dir /data7/emobert/exp/evaluation/${corpus_name_big}/finetune/baseon-taskpretain_movies_v1v2_uniter_4tasks_train400-lr2e5_bs64_max64_train1600
# --checkpoint /data7/emobert/exp/task_pretrain/iemocap_basedon-nomask_movies_v1v2_uniter_4tasks_faceth0.5_mltmelm5_5k-4tasks_maxbb64_faceth0.0_mltmelm5/1/ckpt/model_step_400.pt \
# --output_dir /data7/emobert/exp/evaluation/IEMOCAP/finetune/baseon-taskpretain_movies_v1v2_uniter_4tasks_mltmelm5_train400-lr2e5_bs64_max64_train800/