export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
frozens=$2
dropout=$3

for i in `seq 1 10`;
do
    CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
            --cvNo ${i} --config config/train-emo-iemocap-openface-base-4gpu.json \
            --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_mlm_mrfr_mrckl_3tasks/ckpt/model_step_100000.pt \
            --output_dir /data7/emobert/exp/finetune/iemocap_openface_baseon_nomask_movies_v1_uniter_mlm_mrfr_mrckl_3tasks \
            --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
            --train_batch_size 100
done