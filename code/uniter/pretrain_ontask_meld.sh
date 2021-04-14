export PYTHONPATH=/data7/MEmoBert

gpu_id=$1

## First use trn-data to pretrain on the task dataset
## Then use same trn-data train on downsteam task, train_emo_meld.sh

### 10000 / 128 * 10 = 1000
corpus_name='meld'
for cvNo in `seq 1 1`;
do
        for conth in 0.5;
        do
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
                --cvNo ${cvNo}  --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks.json \
                --model_config config/uniter-base-emoword_nomultitask.json \
                --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_6000.pt \
                --learning_rate 2e-05 --lr_sched_type 'linear'  --gradient_accumulation_steps 2 \
                --conf_th ${conth} --max_bb 36 \
                --train_batch_size 128 --val_batch_size 128 \
                --num_train_steps 2000 --warmup_steps 100 --valid_steps 500 \
                --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-nomask_movies_v1v2v3_uniter_4tasks_faceth${conth}-4tasks_maxbb36_faceth${conth}_trnval/${cvNo}
        done
done

# corpus_name='meld'
# for cvNo in `seq 1 1`;
# do
#         for conth in 0.1 0.5;
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                 --cvNo ${cvNo}  --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks_melm.json \
#                 --melm_prob 0.5 --model_config config/uniter-base-emoword_multitask.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_lr5e5_bs1024_faceth0.5_mlt-melm5/ckpt/model_step_6000.pt \
#                 --learning_rate 2e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 2 \
#                 --conf_th ${conth} --max_bb 36 \
#                 --train_batch_size 128 --val_batch_size 128 \
#                 --num_train_steps 1000 --warmup_steps 0 --valid_steps 200 \
#                 --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-nomask_movies_v1v2v3_uniter_4tasks_faceth0.5_mltmelm5_5k-4tasks_maxbb36_faceth${conth}_mltmelm5/${cvNo}
#         done
# done