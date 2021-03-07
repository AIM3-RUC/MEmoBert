export PYTHONPATH=/data7/MEmoBert
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

# for cvNo in `seq 1 10`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain_ontask.py \
#         --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_faceth0.1/ckpt/model_step_25000.pt \
#         --output_dir /data7/emobert/exp/pretrain/tasks/${corpus_name}_basedon-nomask_movies_v1v2v3_uniter_4tasks_faceth0.1_25k-3tasks/${cvNo} \
#         --cvNo ${cvNo} --learning_rate 1e-5
# done

# corpus_name='msp'
# for cvNo in `seq 6 6`;
# do
# CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain_ontask.py \
#         --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks.json \
#         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_4tasks_faceth0.1/ckpt/model_step_10000.pt \
#         --output_dir /data7/emobert/exp/pretrain/tasks/${corpus_name}_basedon-nomask_movies_v1_uniter_4tasks_faceth0.1_10k-3tasks/${cvNo} \
#         --cvNo ${cvNo} --learning_rate 1e-5
# done

corpus_name='msp'
for cvNo in `seq 5 5`;
do
CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain_ontask.py \
        --config config/pretrain-task-${corpus_name}-base-2gpu_4tasks.json \
        --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_faceth0.1/ckpt/model_step_25000.pt \
        --output_dir /data7/emobert/exp/pretrain/tasks/${corpus_name}_basedon-nomask_movies_v1v2v3_uniter_4tasks_faceth0.1_25k-3tasks/${cvNo} \
        --cvNo ${cvNo} --learning_rate 1e-5
done