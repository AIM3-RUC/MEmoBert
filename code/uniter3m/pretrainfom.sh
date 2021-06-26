export PYTHONPATH=/data7/MEmoBert

### for voxceleb2

## case1: visual + text running on gpu0, only voxceleb2
CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
        --cvNo 0 --n_workers 1  --use_visual  \
        --config config/pretrain-movies-v1v2v3-base-2gpu_visual_text_4tasks_fom_debug.json \
        --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
        --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
        --max_txt_len 30 \
        --IMG_DIM 342 --Speech_DIM 768 \
        --train_batch_size 2 --val_batch_size 2 \
        --num_train_steps 30000 --warmup_steps 3000 --valid_steps 40 \
        --output_dir /data7/emobert/exp/pretrain/nomask_vox2_v1_uniter3m_visual_text_4tasks_fom_debug_lr5e5_bs800_debug