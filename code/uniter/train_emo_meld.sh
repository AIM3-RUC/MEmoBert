export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='meld'

# for meld, directly pretrain, trnsize = 10000/32 * 10 = 3000
# 为了少修改代码，这里将meld放在cvNo=1下面
for frozens in 4 6 8 10; 
do
        for lr in 1e-5 2e-5;
        do
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
                --cls_num 7 \
                --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
                --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
                --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_6000.pt \
                --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
                --learning_rate ${lr} --lr_sched_type 'linear' --conf_th 0.5 \
                --train_batch_size 32 --inf_batch_size 32 --num_train_steps 2000 --valid_steps 200  \
                --output_dir /data7/emobert/exp/evaluation/MELD/finetune/baseon-movies_v1v2v3_uniter_4tasks-lr${lr}_infbs32_th${conf_th}_train3000_trnval_forzen${frozens}
        done
done

# for conf_th in 0.1 0.5; 
# do
#         for lr in 2e-5 5e-5; 
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cls_num 7 \
#                 --cvNo 1 --model_config config/uniter-base-emoword_multitask.json \
#                 --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_lr5e5_bs1024_faceth0.5_mlt-melm5/ckpt/model_step_6000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --conf_th ${conf_th} --max_bb 36 \
#                 --train_batch_size 32 --inf_batch_size 32 --num_train_steps 3000 --valid_steps 300  \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/baseon-movies_v1v2v3_uniter_4tasks_mltmelm5-lr${lr}_infbs32_th${conf_th}_train3000
#         done
# done

### for task pretrained model
# for conf_th in 0.5;
# do
#         for lr in 1e-5 2e-5; 
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cls_num 7 \
#                 --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#                 --checkpoint /data7/emobert/exp/task_pretrain/meld_basedon-nomask_movies_v1v2v3_uniter_4tasks_faceth0.5-4tasks_maxbb36_faceth${conf_th}_trnval/1/ckpt/model_step_1000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --conf_th ${conf_th} \
#                 --train_batch_size 32 --inf_batch_size 32  --num_train_steps 2000 --valid_steps 200 \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/baseon-taskpretrain-movies_v1v2v3_uniter_4tasks_faceth0.5_step1000-lr${lr}_infbs32_faceth${conf_th}_trnval
#         done
# done

# for conf_th in 0.1 0.5;
# do
#         for lr in 1e-5 2e-5 5e-5; 
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cls_num 7 \
#                 --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#                 --checkpoint /data7/emobert/exp/task_pretrain/meld_basedon-nomask_movies_v1v2v3_uniter_4tasks_faceth0.5_mltmelm5_5k-4tasks_maxbb36_faceth${conf_th}_mltmelm5/1/ckpt/model_step_1000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --conf_th ${conf_th} \
#                 --train_batch_size 32 --inf_batch_size 32   --num_train_steps 3000 \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/baseon-taskpretrain-movies_v1v2v3_uniter_4tasks_faceth0.5_mltmelm5_step1000-lr${lr}_infbs32_faceth${conf_th}
#         done
# done