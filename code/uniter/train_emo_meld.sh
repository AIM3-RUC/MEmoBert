export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='meld'

# for meld, directly pretrain, trnsize = 10000/32 * 10 = 3000
# 为了少修改代码，这里将meld放在cvNo=1下面
# for frozens in 4 6 8 10; 
# do
#         for lr in 1e-5 2e-5;
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cls_num 7 \
#                 --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_6000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --conf_th 0.5 \
#                 --train_batch_size 32 --inf_batch_size 32 --num_train_steps 2000 --valid_steps 200  \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/baseon-movies_v1v2v3_uniter_4tasks-lr${lr}_infbs32_th${conf_th}_train3000_trnval_forzen${frozens}
#         done
# done


# for frozens in 4 6 8 10; 
# do
#         for lr in 1e-5 2e-5;
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cls_num 7 \
#                 --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#                 --checkpoint /data7/emobert/exp/task_pretrain/meld_basedon-nomask_movies_v1v2v3_uniter_4tasks_faceth0.5-4tasks_maxbb36_faceth0.5_trnval/1/ckpt/model_step_1000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --conf_th 0.5 \
#                 --train_batch_size 32 --inf_batch_size 32 --num_train_steps 2000 --valid_steps 200  \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/baseon-taskpretrain-movies_v1v2v3_uniter_4tasks-lr${lr}_infbs32_th0.5_train2000_trnval_forzen${frozens}
#         done
# done

### for direct train
# for conf_th in 0.5;
# do
#         for lr in 2e-5 5e-5 1e-4; 
#         do
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cls_num 7 \
#                 --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#                 --checkpoint /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --conf_th ${conf_th} \
#                 --train_batch_size 32 --inf_batch_size 32  --num_train_steps 3000 --valid_steps 300 \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/baseon-direct-lr${lr}_infbs32_faceth${conf_th}_trnval
#         done
# done

# for high-quality
for frozens in 0; 
do
        for lr in 5e-5;
        do
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
                --cls_num 7 \
                --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
                --config config/train-emo-${corpus_name}-openface-hq-base-2gpu.json \
                --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_6000.pt \
                --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
                --learning_rate ${lr} --lr_sched_type 'linear' --conf_th 0.5 \
                --train_batch_size 2 --inf_batch_size 2 --num_train_steps 2000 --valid_steps 200  \
                --output_dir /data7/emobert/exp/evaluation/MELD/finetune/baseon-movies_v1v2v3_uniter_4tasks-lr${lr}_infbs32_th0.5_train2000_trnval_hq
        done
done
