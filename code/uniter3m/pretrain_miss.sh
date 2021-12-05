source deactivate base
export PYTHONPATH=/data7/MEmoBert

## 可以同时利用单模态，或者任意模态的组合进行训练，注意此时预训练任务不能 --use_visual 来进行判断，而是config中每个db模态信息是否存在.

## case1: speech + visual + text running on wwm + span - itm + cases
# CUDA_VISIBLE_DEVICES=2 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_7cases_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_7cases_noitm_lr5e5_bs800

## case2: speech + visual + text running on wwm + span - itm + cases + morelargespan
# CUDA_VISIBLE_DEVICES=3 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4 --use_visual --use_speech \
#         --mask_speech_consecutive 5 --mask_speech_len_ratio 0.5 --mask_visual_consecutive 5 --mask_visual_len_ratio 0.5 \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_speechwav2vec_5tasks_wwm_span_7cases_noitm.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 160 --val_batch_size 160 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_7cases_noitm_mores5.5v5.5_lr5e5_bs800