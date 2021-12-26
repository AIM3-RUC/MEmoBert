
source deactivate base
export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='iemocap'
corpus_name_L='IEMOCAP'

## Outline
################# Part0: Baselines- DirectTrain + Finetune task ###################################################
################# Part1: Explore the ITM task ###################################################
################# Part2: Explore WWM + Span tasks ###################################################
################# Part3: Explore WWM + Span tasks + cross-modality-prompt ###################################################
################# Part4: Explore WWM + Span tasks + onlylva ###################################################
################# Part9: ablation, Explore mlm + mrkcl + mrfr tasks + prompt ###################################################
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_speech_text_5tasks_noitm_lr5e5_bs800/ckpt/model_step_30000.pt
################# Part10: ablation, pretain without span-visual, span-acoustic + prompt ###################################################
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_noaudio_lr5e5_bs800/ckpt/model_step_40000.pt
################# Part11: ablation, pretain without span-visual, span-acoustic + prompt ###################################################
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_novisual_lr5e5_bs800/ckpt/model_step_40000.pt
################# Part12: ablation, pretain with more efficient audio and visual masking strategies + prompt ###################################################
################# Part13: Flexiable Prompt and softprompt for lva modality########################
################# Part14: compare different masking strategies on cm ########################
################# Part15: compare different masking strategies on lva ########################


################# Part0: Baselines- DirectTrain + Finetune task ################################################### 
# for lr in 5e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         num_train_steps=2000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_speech --use_visual --use_text \
#                 --seed ${seed} \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_trn.json \
#                 --checkpoint /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-directTrain-uniter3m_visual_wav2vec_text-lr${lr}_train${num_train_steps}_trn_seed${seed}
#         done
#     done
# done

# for lr in 5e-5
# do
#     for seed in 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         num_train_steps=2000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_speech --use_visual --use_text \
#                 --seed ${seed} \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-directTrain-FromScratch-uniter3m_visual_wav2vec_text-lr${lr}_train${num_train_steps}_trnval_seed${seed}
#         done
#     done
# done

#### case2:text + speech + visual + wwm + span -itm
# for lr in 5e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         num_train_steps=1500
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                 --seed ${seed} \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_trn.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-lr${lr}_train${num_train_steps}_trn_seed${seed}
#         done
#     done
# done

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
#icassp setting
# for lr in 3e-5
# do
#     for seed in 42 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         prompt_type=iam
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type ${prompt_type} --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_icassp_7cases.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 4000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt${prompt_type}_icassp_7cases_lr${lr}_seed${seed}/${cvNo}
#         done
#     done
# done

################# Part4: Explore WWM + Span tasks + onlylva ###################################################
#icassp onlylva:---
# for lr in 3e-5
# do
#     for seed in 42 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         prompt_type=iam
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type ${prompt_type} --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_icassp_onlylva.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt${prompt_type}_icassp_onlylva_lr${lr}_seed${seed}/${cvNo}
#         done
#     done
# done

################# Part9: ablation, Explore mlm + mrkcl + mrfr tasks + prompt ###################################################
# for lr in 3e-5 5e-5
# do
#     for seed in 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_speech_text_5tasks_noitm_lr5e5_bs800/ckpt/model_step_30000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_speech_text_5tasks_noitm-cm_mask_prompt_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

################# Part10: ablation, pretain without span-visual, span-acoustic + prompt ###################################################
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_noaudio_lr5e5_bs800/ckpt/model_step_40000.pt
# for lr in 5e-5
# do
#     for seed in 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_noaudio_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_speech_text_5tasks_wwm_span_noitm_noaudio-cm_mask_prompt_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

################# Part11: ablation, pretain without span-visual, span-acoustic + prompt ###################################################
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_novisual_lr5e5_bs800/ckpt/model_step_40000.pt
# for lr in 5e-5
# do
#     for seed in 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_novisual_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_speech_text_5tasks_wwm_span_noitm_novisual-cm_mask_prompt_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

################# Part12: ablation, pretain with more efficient audio and visual masking strategies + prompt ###################################################
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_spanS5.5V5.5_lr5e5_bs800/ckpt/model_step_30000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_spanS7.5V5.5_lr5e5_bs800/ckpt/model_step_30000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_mores5.5v5.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.3s.3_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.5s.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.7s.7_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 

# case1: all seven cases spanS5.5V5.5
# for lr in 3e-5
# do
#     for seed in 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_icassp.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_spanS5.5V5.5_lr5e5_bs800/ckpt/model_step_30000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_spanS5.5V5.5_step3w-cm_mask_prompt_onlylva_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

# case1: all seven cases mores5.5v5.5
# for lr in 3e-5
# do
#     for seed in 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_icassp.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_mores5.5v5.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_mores5.5v5.5_noitm_step4w-cm_mask_prompt_icassp_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

# case2: all seven cases spanS7.5V5.5
# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_onlylva.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_spanS7.5V5.5_lr5e5_bs800/ckpt/model_step_30000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_spanS7.5V5.5_step3w-cm_mask_prompt_onlylva_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

# case3: all seven cases mrm_msrm_maskprobv.3s.3
# for lr in 3e-5
# do
#     for seed in 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_onlylva.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.3s.3_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_mrm_msrm_maskprobv.3s.3_step4w-cm_mask_prompt_onlylva_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

# case4: all seven cases mrm_msrm_maskprobv.5s.5
# for lr in 5e-5
# do
#     for seed in 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_onlylva.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.5s.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_mrm_msrm_maskprobv.5s.5_step4w-cm_mask_prompt_onlylva_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

# # case5: all seven cases mrm_msrm_maskprobv.7s.7
# for lr in 5e-5
# do
#     for seed in 5678
#     do
#         for cvNo in $(seq 6 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.7s.7_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_mrm_msrm_maskprobv.7s.7_step4w-cm_mask_prompt_onlylva_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

# ################# Part13: Flexiable Prompt and softprompt for lva modality########################
# for lr in 5e-5
# do
#     for seed in 5678
#     do
#         for cvNo in $(seq 8 10)
#         do
#         prompt_type=iam
#         num_train_steps=2500
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type ${prompt_type} --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_flexprompt_onlylva.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps ${num_train_steps} --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_flexprompt${prompt_type}_onlylva_lr${lr}_trn${num_train_steps}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

# for lr in 5e-5
# do
#     for seed in 5678
#     do
#         for cvNo in $(seq 8 10)
#         do
#         prompt_type=softprompt5
#         num_train_steps=2500
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type ${prompt_type} --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_flexprompt_onlylva.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps ${num_train_steps} --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_flexprompt${prompt_type}_onlylva_lr${lr}_trn${num_train_steps}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

# for lr in 5e-5
# do
#     for seed in 5678
#     do
#         for cvNo in $(seq 8 10)
#         do
#         prompt_type=halfsoftprompt5iam
#         num_train_steps=2500
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type ${prompt_type} --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_flexprompt_onlylva.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps ${num_train_steps} --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_flexprompt${prompt_type}_onlylva_lr${lr}_trn${num_train_steps}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

################# Part14: compare different masking strategies on cm ########################
# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_icassp.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_lr5e5_bs800/ckpt/model_step_60000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_step6w-cm_mask_prompt_icassp_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_icassp.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_step4w-cm_mask_prompt_icassp_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

################# Part15: compare different masking strategies on lva ########################
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_onespans.5v.5_noitm_lr5e5_bs800//ckpt/model_step_40000.pt \

# masking 1.0 
# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_icassp.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 2500 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_step4w-cm_mask_prompt_icassp_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_icassp_onlylva.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_onespans.1v.1_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 2500 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_onespans.1v.1_noitm_step4w-cm_mask_prompt_icassp_onlylva_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done