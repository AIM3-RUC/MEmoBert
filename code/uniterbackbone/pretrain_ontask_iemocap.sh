export PYTHONPATH=/data7/MEmoBert

gpu_id=$1
corpus_name='iemocap'

## First use trn-data to pretrain on the task dataset
## Then use same trn-data train on downsteam task 

# iemocap maxbb=64, conf_th=0.0, 4500 / 128  * 10 = 400
for maxbb in 36 64;
do
    for cvNo in `seq 1 10`;
    do
    CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
            --cvNo ${cvNo}  --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks.json \
            --melm_prob 0.5 --model_config config/uniter-base-emoword_nomultitask.json \
            --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_6000.pt \
            --learning_rate 2e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
            --conf_th 0.0 --max_bb ${maxbb} \
            --train_batch_size 64 --val_batch_size 64 \
            --num_train_steps 500 --warmup_steps 0 --valid_steps 100 \
            --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-nomask_movies_v1v2v3_uniter_4tasks_faceth0.5_5k-4tasks_maxbb${maxbb}_faceth0.0_train4k/${cvNo}
    done
done

# for maxbb in 36 64;
# do
#     for cvNo in `seq 1 10`;
#     do
#     CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#             --cvNo ${cvNo}  --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks_melm.json \
#             --melm_prob 0.5 --model_config config/uniter-base-emoword_multitask.json \
#             --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_lr5e5_bs1024_faceth0.5_mlt-melm5/ckpt/model_step_6000.pt \
#             --learning_rate 2e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#             --conf_th 0.0 --max_bb ${maxbb} \
#             --train_batch_size 64 --val_batch_size 64 \
#             --num_train_steps 400 --warmup_steps 0 --valid_steps 100 \
#             --output_dir /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-nomask_movies_v1v2v3_uniter_4tasks_faceth0.5_mltmelm5_5k-4tasks_maxbb${maxbb}_faceth0.0_mltmelm5/${cvNo}
#     done
# done