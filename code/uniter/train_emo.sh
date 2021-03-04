export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
frozens=$2
dropout=$3
corpus_name='msp'
for i in `seq 1 12`;
do
    CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
            --cvNo ${i} --config config/train-emo-${corpus_name}-openface-base-4gpu.json \
            --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_4tasks_faceth0.1/ckpt/model_step_10000.pt \
            --output_dir /data7/emobert/exp/finetune/${corpus_name}_openface_baseon_nomask_movies_v1_uniter_4tasks_10k_fix5e5 \
            --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
            --learning_rate 5e-5 --train_batch_size 80 
done

for i in `seq 1 12`;
do
    CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
            --cvNo ${i} --config config/train-emo-${corpus_name}-openface-base-4gpu.json \
            --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_4tasks_faceth0.1/ckpt/model_step_50000.pt \
            --output_dir /data7/emobert/exp/finetune/${corpus_name}_openface_baseon_nomask_movies_v1_uniter_4tasks_50k_fix5e5 \
            --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
            --learning_rate 5e-5 --train_batch_size 80 
done

for i in `seq 1 12`;
do
    CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
            --cvNo ${i} --config config/train-emo-${corpus_name}-openface-base-4gpu.json \
            --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2_uniter_4tasks_faceth0.1/ckpt/model_step_20000.pt \
            --output_dir /data7/emobert/exp/finetune/${corpus_name}_openface_baseon_nomask_movies_v1v2_uniter_4tasks_20k_fix5e5 \
            --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
            --learning_rate 5e-5 --train_batch_size 80 
done

for i in `seq 1 12`;
do
    CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
            --cvNo ${i} --config config/train-emo-${corpus_name}-openface-base-4gpu.json \
            --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2_uniter_4tasks_faceth0.1/ckpt/model_step_45000.pt \
            --output_dir /data7/emobert/exp/finetune/${corpus_name}_openface_baseon_nomask_movies_v1v2_uniter_4tasks_45k_fix5e5 \
            --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
            --learning_rate 5e-5 --train_batch_size 80 
done

for i in `seq 1 12`;
do
    CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
            --cvNo ${i} --config config/train-emo-${corpus_name}-openface-base-4gpu.json \
            --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_faceth0.1/ckpt/model_step_25000.pt \
            --output_dir /data7/emobert/exp/finetune/${corpus_name}_openface_baseon_nomask_movies_v1v2v3_uniter_4tasks_25k_fix5e5 \
            --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
            --learning_rate 5e-5 --train_batch_size 80 
done

for i in `seq 1 12`;
do
    CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
            --cvNo ${i} --config config/train-emo-${corpus_name}-openface-base-4gpu.json \
            --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter_4tasks_faceth0.1/ckpt/model_step_45000.pt \
            --output_dir /data7/emobert/exp/finetune/${corpus_name}_openface_baseon_nomask_movies_v1v2v3_uniter_4tasks_45k_fix5e5 \
            --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
            --learning_rate 5e-5 --train_batch_size 80 
done