export PYTHONPATH=/data7/MEmoBert

## 可以同时利用单模态，或者任意模态的组合进行训练，注意此时预训练任务不能 --use_visual 来进行判断，而是config中每个db模态信息是否存在.

## case1: visual + text running on onlytext is 5corpus_emo4 based on 4tasks+emocls ablation
# CUDA_VISIBLE_DEVICES=7 horovodrun -np 1 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_visual_text_4tasks_indemocls_5corpus_emo5.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 100 --val_batch_size 100 \
#         --num_train_steps 20000 --warmup_steps 2000 --valid_steps 5000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_indemocls_5corpus_emo5_lr5e5_bs512

## case2: visual + text running on onlytext is opensub p1p2 based on 4tasks+emocls ablation
CUDA_VISIBLE_DEVICES=6,7 horovodrun -np 2 python pretrain.py \
        --cvNo 0 --n_workers 4  --use_visual  \
        --config config/pretrain-movies-v1v2v3-base-2gpu_visual_text_4tasks_indemocls_p1p2_emo5.json \
        --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
        --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
        --IMG_DIM 342 --Speech_DIM 768 \
        --train_batch_size 200 --val_batch_size 200 \
        --num_train_steps 40000 --warmup_steps 4000 --valid_steps 8000 \
        --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_indemocls_p1p2_emo5_lr5e5_bs512

## case3: visual + text + speech running on onlytext is opensub p1p2 based on 4tasks+emocls ablation
# CUDA_VISIBLE_DEVICES=0,1 horovodrun -np 2 python pretrain.py \
#         --cvNo 0 --n_workers 4  --use_visual  \
#         --config config/pretrain-movies-v1v2v3-base-2gpu_visual_text_4tasks_indemocls_p1p2_emo5.json \
#         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#         --learning_rate 5e-05 --lr_sched_type 'linear' --gradient_accumulation_steps 4 \
#         --IMG_DIM 342 --Speech_DIM 768 \
#         --train_batch_size 100 --val_batch_size 100 \
#         --num_train_steps 40000 --warmup_steps 4000 --valid_steps 8000 \
#         --output_dir /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_indemocls_p1p2_emo5_lr5e5_bs512