export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
frozens=$2
dropout=$3
CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
        --cv_no 1 --config config/train-emo-msp-base-4gpu.json \
        --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_mlm_mrfr_mrckl_3tasks/ckpt/model_step_100000.pt \
        --output_dir /data7/emobert/exp/finetune/msp_baseon_nomask_movies_v1_uniter_mlm_mrfr_mrckl_3tasks \
        --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type emocls --postfix none \
        --train_batch_size 64