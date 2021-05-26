export PYTHONPATH=/data7/MEmoBert

### for voxceleb2

## case1: visual + text running on gpu0
CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
        --cvNo 0 --n_workers 4  --use_visual  \
        --config config/pretrain-vox2-v1-base-2gpu_4tasks_emo_sentiword.json \
        --model_config config/uniter-base-emoword_nomultitask_difftype.json \
        --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
        --max_txt_len 60 \
        --IMG_DIM 342 --Speech_DIM 768 \
        --train_batch_size 20 --val_batch_size 20 \
        --num_train_steps 20000 --warmup_steps 2000 --valid_steps 20 \
        --output_dir /data7/emobert/exp/pretrain/nomask_vox2_v1_uniter3m_visual_text_4tasks_lr5e5_bs512_faceth0.5