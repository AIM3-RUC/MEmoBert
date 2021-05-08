export PYTHONPATH=/data7/MEmoBert

### 在当前代码中，text+visual是正常.
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_4tasks_emo.json \
#         --model_config config/uniter-base-emoword_multitask.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --melm_prob 0.5 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
#         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 2000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_lr5e5_faceth0.5_melm.5

# CUDA_VISIBLE_DEVICES=2,5 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_4tasks_emo_sentiword.json \
#         --model_config config/uniter-base-emoword_multitask.json \
#         --learning_rate 5e-5 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --melm_prob 0.5 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --conf_th 0.5 --max_txt_len 30 --max_bb 36 \
#         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#         --train_batch_size 64 --val_batch_size 64 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 2000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_lr5e5_faceth0.5_melm.5_sentiword