export PYTHONPATH=/data4/MEmoBert
gpu_id=$1
# corpus_name='iemocap'
# for cvNo in `seq 2 10`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain_ontask.py \
#         --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_4tasks_faceth0.1/ckpt/model_step_10000.pt \
#         --output_dir /data7/emobert/exp/pretrain/tasks/${corpus_name}_basedon-nomask_movies_v1_uniter_4tasks_faceth0.1_10k-3tasks/${cvNo} \
#         --cvNo ${cvNo} --learning_rate 1e-5
# done

# corpus_name='msp'
# for cvNo in `seq 5 5`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain_ontask.py \
#         --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_faceth0.1/ckpt/model_step_25000.pt \
#         --output_dir /data7/emobert/exp/pretrain/tasks/${corpus_name}_basedon-nomask_movies_v1v2v3_uniter_4tasks_faceth0.1_25k-3tasks/${cvNo} \
#         --cvNo ${cvNo} --learning_rate 1e-5
# done

# 10000 / 128 * 6 = 600, meld 的 soft-label 不对 --pending
corpus_name='meld'
for cvNo in `seq 1 1`;
do
CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain_ontask.py \
        --cvNo ${cvNo}  --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks.json \
        --model_config config/uniter-base-emoword_nomultitask.json \
        --checkpoint /data4/emobert/exp/pretrain/nomask_movies_v1v2_uniter_4tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_5000.pt \
        --conf_th 0.1 --max_bb 36 \
        --learning_rate 2e-05 --gradient_accumulation_steps 2 
        --train_batch_size 64 --val_batch_size 64 --num_train_steps 600 \
        --output_dir /data4/emobert/exp/task_pretrain/${corpus_name}_basedon-nomask_movies_v1v2_uniter_4tasks_faceth0.5_5k-4tasks_maxbb36_faceth0.1/${cvNo}
done