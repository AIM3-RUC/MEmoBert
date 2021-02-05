export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
frozens=$2
dropout=$3

corpus_name='iemocap'

for i in `seq 8 10`;
do
    CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
            --cvNo ${i} --config config/train-emo-${corpus_name}-openface-base-4gpu.json \
            --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_4tasks_melm0.3/ckpt/model_step_100000.pt \
            --output_dir /data7/emobert/exp/finetune/${corpus_name}_openface_baseon_nomask_movies_v1_uniter_4tasks_melm0.3 \
            --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
            --train_batch_size 80
done