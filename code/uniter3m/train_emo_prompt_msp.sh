
source deactivate base
export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='msp'
corpus_name_L='MSP'

## Outline
################# Part1: Explore the ITM task ################################################### 
################# Part2: Explore WWM + Span tasks ################################################### 
################# Part3: Explore WWM + Span - ITM tasks + cross-modality-prompt ################################################### 

################# Part1: Explore the ITM task ################################################### 
# for lr in 5e-5
# do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                 --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                 --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_nsp_iam_prompt.json \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_lr5e5_bs800/ckpt/model_step_40000.pt \
#                 --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size 32 --val_batch_size 32 \
#                 --num_train_steps 2000 --warmup_steps 100 --valid_steps 100 \
#                 --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_step4w-nsp_iam_prompt_lr${lr}_trnval/${cvNo}

#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                 --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                 --seed 4321 \
#                 --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_nsp_iam_prompt.json \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_lr5e5_bs800/ckpt/model_step_40000.pt \
#                 --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size 32 --val_batch_size 32 \
#                 --num_train_steps 2000 --warmup_steps 100 --valid_steps 100 \
#                 --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_step4w-nsp_iam_prompt_diffseed_lr${lr}_trnval/${cvNo}
#         done
# done

################# Part2: Explore WWM + Span tasks ################################################### 
# case1: taskptrain visual + speech + text - itm + wwm + span
# for lr in 5e-5
# do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_mask_iam_prompt.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 2000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-mask_iam_prompt_lr${lr}_trnval/${cvNo}
#         # differ seed
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --seed 4321 \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_mask_iam_prompt.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 2000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-mask_iam_prompt_diffseed_lr${lr}_trnval/${cvNo}
#         done
# done

# for lr in 5e-5
# do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                 --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                 --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_nsp_iam_prompt.json \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                 --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size 32 --val_batch_size 32 \
#                 --num_train_steps 2000 --warmup_steps 100 --valid_steps 100 \
#                 --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-nsp_iam_prompt_lr${lr}_trnval/${cvNo}

#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                 --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                 --seed 4321 \
#                 --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_nsp_iam_prompt.json \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                 --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size 32 --val_batch_size 32 \
#                 --num_train_steps 2000 --warmup_steps 100 --valid_steps 100 \
#                 --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-nsp_iam_prompt_diffseed_lr${lr}_trnval/${cvNo}
#         done
# done

################# Part3: Explore WWM + Span - ITM tasks + cross-modality-prompt ################################################### 
# case1: only mask-va mask-v mask-a
# for lr in 5e-5
# do
#         for cvNo in $(seq 1 12)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_onlycm.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_onlycm_lr${lr}_trnval/${cvNo}
#         done
# done

# # case2: only l_mask-va l-mask-v l-mask-a l-mask
# for lr in 5e-5
# do
#         for cvNo in $(seq 1 12)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_nocm.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_nocm_lr${lr}_trnval/${cvNo}
#         done
# done

# case3: all seven cases
# for lr in 5e-5
# do
#         for cvNo in $(seq 1 12)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 5000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_lr${lr}_trnval/${cvNo}
#         done
# done

# melm-based model
for lr in 3e-5 5e-5
do
        for cvNo in $(seq 1 12)
        do
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
                        --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
                        --prompt_type iam \
                        --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt.json \
                        --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
                        --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_speech_text_5tasks_melm_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_30000.pt \
                        --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
                        --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
                        --train_batch_size 32 --val_batch_size 32 \
                        --num_train_steps 4000 --warmup_steps 100 --valid_steps 100 \
                        --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_melm_wwm_span_noitm_step3w-cm_mask_prompt_lr${lr}_trnval/${cvNo}
        done
done