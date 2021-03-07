export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
frozens=$2
dropout=$3
corpus_name='msp'
for i in `seq 5 5`;
do
    CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
            --cvNo ${i} --config config/train-emo-${corpus_name}-openface-base-4gpu.json \
            --checkpoint /data7/emobert/exp/pretrain/tasks/${corpus_name}_basedon-nomask_movies_v1v2v3_uniter_4tasks_faceth0.1_25k-3tasks/${i}/ckpt/model_step_1000.pt \
            --output_dir /data7/emobert/exp/finetune/${corpus_name}_basedon-nomask_movies_v1v2v3_uniter_4tasks_faceth0.1_25k_3tasks-fix5e5  \
            --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
            --learning_rate 5e-5 --train_batch_size 80 
done