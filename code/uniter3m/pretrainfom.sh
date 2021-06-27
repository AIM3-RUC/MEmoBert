export PYTHONPATH=/data7/MEmoBert

# ## case1: visual + text 
# CUDA_VISIBLE_DEVICES=0 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_visual_text_4tasks_vfom.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 30 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_vfom_lr5e5_bs800

# ## case2: speech + text 
# CUDA_VISIBLE_DEVICES=1 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_3tasks_sfom.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --max_txt_len 30 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 200 --val_batch_size 200 \
#         --num_train_steps 30000 --warmup_steps 3000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speech_text_3tasks_sfom_lr5e5_bs800

## case3: speech + text + visual
CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python pretrain.py \
        --cvNo 0 --n_workers 4  --use_speech --use_visual \
        --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_vfom_sfom.json \
        --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
        --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
        --max_txt_len 30 \
        --IMG_DIM 342 --Speech_DIM 768 \
        --train_batch_size 160 --val_batch_size 160 \
        --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
        --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speech_visual_text_5tasks_vfom_sfom_lr5e5_bs800