export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='msp'
corpus_name_big='MSP'

# for msp, trnsize = 3500/32 * 10 = 1200
# for frozens in 8 10;
# do
#         for lr in 2e-5 5e-5;
#         do
#                 for cvNo in `seq 1 12`;
#                 do
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                         --cvNo ${cvNo} --model_config config/uniter-base-emoword_nomultitask.json \
#                         --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_6000.pt \
#                         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                         --learning_rate ${lr} --lr_sched_type 'linear' --conf_th 0.0 --max_bb 36 \
#                         --train_batch_size 32 --inf_batch_size 32 --num_train_steps 1000 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/evaluation/${corpus_name_big}/finetune/baseon-movies_v1v2v3_uniter_4tasks-lr${lr}_bs32_max36_train1200_trnval_forzen${frozens}
#                 done
#         done
# done


# for frozens in 4 6 8 10;
# do
#         for lr in 2e-5 5e-5;
#         do
#                 for cvNo in `seq 1 12`;
#                 do
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                         --cvNo ${cvNo} --model_config config/uniter-base-emoword_nomultitask.json \
#                         --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
#                         --checkpoint /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-nomask_movies_v1v2v3_uniter_4tasks_faceth0.5_5k-4tasks_maxbb36_faceth0.0_trnval/${cvNo}/ckpt/model_step_1000.pt \
#                         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                         --learning_rate ${lr} --lr_sched_type 'linear' --conf_th 0.0 --max_bb 36 \
#                         --train_batch_size 32 --inf_batch_size 32 --num_train_steps 1000 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/evaluation/${corpus_name_big}/finetune/baseon-taskpretain-movies_v1v2v3_uniter_4tasks-lr${lr}_bs32_max36_train1200_trnval_forzen${frozens}
#                 done
#         done
# done

for conf_th in 0.0;
do
        for lr in 1e-4; 
        do
        for cvNo in `seq 1 12`;
                do
                frozens=0
                CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
                        --cls_num 4 \
                        --cvNo ${cvNo} --model_config config/uniter-base-emoword_nomultitask.json \
                        --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
                        --checkpoint /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
                        --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
                        --learning_rate ${lr} --lr_sched_type 'linear' --conf_th ${conf_th} \
                        --train_batch_size 32 --inf_batch_size 32  --num_train_steps 3000 --valid_steps 300 \
                        --output_dir /data7/emobert/exp/evaluation/${corpus_name_big}/finetune/baseon-direct-lr${lr}_infbs32_faceth${conf_th}_trnval
                done
        done
done