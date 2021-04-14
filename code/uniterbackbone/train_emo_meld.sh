export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
frozens=0
dropout=0.1
corpus_name='meld'

# 关于模型的初始化，同时加载backbone和Uniter, 默认是所有模块全部进行finetune.

# for meld, directly pretrain, trnsize = 10000/32 * 10 = 3000
# for conf_th in 0.5;
# do
#         for lr in 1e-5 2e-5;
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cls_num 7 \
#                 --cvNo 1 --model_config config/uniter-base-backbone_3dresnet.json \
#                 --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#                 --checkpoint /data7/emobert/exp/pretrain/resnet3d_nomask_movies_v1v2v3_uniter_mlmitm_lr5e5-backbone_scratch_optimFalse-bs480_faceth0.5_continue1w/ckpt/model_step_10000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --conf_th ${conf_th} --max_bb 36 \
#                 --train_batch_size 32 --inf_batch_size 32 --num_train_steps 2000 --valid_steps 200  \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/resnet3d_baseon-movies_v1v2v3_uniter_lr5e5-backbone_scratch_optimFalse_continue1w-lr${lr}_th${conf_th}_max36_train3000_trnval
#         done    
# done

# for task-pretrain
for conf_th in 0.5;
do
        for lr in 1e-5 2e-5;
        do
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
                --cls_num 7 \
                --cvNo 1 --model_config config/uniter-base-backbone_3dresnet.json \
                --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
                --checkpoint /data7/emobert/exp/task_pretrain/meld_basedon-resnet3d_nomask_movies_v1v2v3_uniter_lr5e5-backbone_scratch_optimFalse_continue1w-2tasks_maxbb36_faceth0.5_trnval/1/ckpt/model_step_2000.pt \
                --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
                --learning_rate ${lr} --lr_sched_type 'linear' --conf_th ${conf_th} --max_bb 36 \
                --train_batch_size 32 --inf_batch_size 32 --num_train_steps 2000 --valid_steps 200  \
                --output_dir /data7/emobert/exp/evaluation/MELD/finetune/resnet3d_taskpretrain-baseon-movies_v1v2v3_uniter_lr5e5_backbone_scratch_optimFalse_continue1w_trnval-lr${lr}_th${conf_th}_max36_train3000_trnval
        done    
done