export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
frozens=0
dropout=0.1
corpus_name='iemocap'

# 关于模型的初始化，同时加载backbone和Uniter, 默认是所有模块全部进行finetune.

# for iemocap, directly pretrain, trnsize = 4000/32 * 10 = 1000, if maxbb=36, batchsize=32, if maxbb=64 then batchsize=16, step=5000
# for max_bb in 36;
# do
#         for lr in 5e-5;
#         do
#                 for cvNo in `seq 1 10`;
#                 do
#                 if [ ${max_bb} == 36 ]; then
#                         train_batch_size=32
#                         inf_batch_size=1
#                         num_train_steps=1500
#                 else
#                         train_batch_size=16
#                         inf_batch_size=1
#                         num_train_steps=2000
#                 fi
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                         --cls_num 7 \
#                         --cvNo ${cvNo} --model_config config/uniter-base-backbone_3dresnet.json \
#                         --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#                         --checkpoint /data7/emobert/exp/pretrain/resnet3d_nomask_movies_v1v2v3_uniter_mlmitm_lr5e5-backbone_scratch_optimFalse-bs480_faceth0.5_continue1w/ckpt/model_step_10000.pt \
#                         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
#                         --learning_rate ${lr} --lr_sched_type 'linear' --conf_th 0.0 --max_bb ${max_bb} \
#                         --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} --num_train_steps ${num_train_steps} --valid_steps 200  \
#                         --output_dir /data7/emobert/exp/evaluation/IEMOCAP/finetune/resnet3d_baseon-movies_v1v2v3_uniter_lr5e5-backbone_scratch_optimFalse_continue1w-lr${lr}_bs${train_batch_size}infbs${inf_batch_size}_th0.0_max${max_bb}_train${num_train_steps}
#                 done
#         done
# done

for max_bb in 36;
do
        for lr in 5e-5;
        do
                for cvNo in `seq 1 10`;
                do
                if [ ${max_bb} == 36 ]; then
                        train_batch_size=32
                        inf_batch_size=1
                        num_train_steps=1400
                else
                        train_batch_size=16
                        inf_batch_size=1
                        num_train_steps=4000
                fi
                CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
                        --cls_num 7 \
                        --cvNo ${cvNo} --model_config config/uniter-base-backbone_3dresnet.json \
                        --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
                        --checkpoint /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-resnet3d_nomask_movies_v1v2v3_uniter_lr5e5-backbone_scratch_optimFalse_continue1w-2tasks_maxbb36_faceth0.0/${cvNo}/ckpt/model_step_1500.pt \
                        --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
                        --learning_rate ${lr} --lr_sched_type 'linear' --conf_th 0.0 --max_bb ${max_bb} \
                        --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} --num_train_steps ${num_train_steps} --valid_steps 200  \
                        --output_dir /data7/emobert/exp/evaluation/IEMOCAP/finetune/resnet3d_taskpretrain-baseon-movies_v1v2v3_uniter_lr5e5_backbone_scratch_optimFalse_continue1w-lr${lr}_bs${train_batch_size}infbs${inf_batch_size}_th0.0_max${max_bb}_train${num_train_steps}
                done
        done
done