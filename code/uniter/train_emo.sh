export PYTHONPATH=/data7/MEmoBert
CUDA_VISIBLE_DEVICES=5 horovodrun -np 1 python train_emo.py \
        --cv_no 1 --config config/train-emo-iemocap-base-4gpu.json \
        --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1_uniter_mlm_itm_2tasks/ckpt/model_step_100000.pt \
        --output_dir /data7/emobert/exp/finetune/iemocap_baseon_nomask_movies_v1_uniter_mlm_itm_2tasks \
        --frozen_en_layers 7 --cls_dropout 0.3 --cls_type emocls --postfix none \
        --train_batch_size 128