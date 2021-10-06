
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
################# Part4: Explore WWM + Span tasks + TaskPretain + Finetune ###################################################
################# Part5: Explore WWM + Span tasks + TaskPretain + cross-modality prompt ###################################################
################# Part6: few-shot of directly train ###################################################
################# Part7: few-shot of finetune ###################################################
################# Part8: few-shot of prompt + seven augment ###################################################
################# Part9: ablation, Explore mlm + mrkcl + mrfr tasks + prompt ###################################################
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_speech_text_5tasks_noitm_lr5e5_bs800/ckpt/model_step_30000.pt
################# Part10: ablation, pretain without span-visual, span-acoustic + prompt ###################################################
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_noaudio_lr5e5_bs800/ckpt/model_step_40000.pt
################# Part11: ablation, pretain without span-visual, span-acoustic + prompt ###################################################
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_novisual_lr5e5_bs800/ckpt/model_step_40000.pt

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
# case1: only mask-va mask-v mask-a
# for lr in 5e-5
# do
#         for cvNo in $(seq 1 10)
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
#                         --num_train_steps 5000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_onlycm_lr${lr}_trnval/${cvNo}
#         done
# done

# case2: only l_mask-va l-mask-v l-mask-a l-mask
# for lr in 5e-5
# do
#         for cvNo in $(seq 1 10)
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
#                         --num_train_steps 4000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_nocm_lr${lr}_trnval/${cvNo}
#         done
# done

# case1: all seven cases
# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_trn.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 4000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_lr${lr}_trn_seed${seed}/${cvNo}
#         done
#     done
# done

# case1: only lva cases
# for lr in 5e-5
# do
#     for seed in 4321
#     do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_lva_trn.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 4000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_lr${lr}_lva_trn_seed${seed}/${cvNo}
#         done
#     done
# done

# # melm-based model
# for lr in 3e-5 5e-5
# do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_speech_text_5tasks_melm_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_30000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 4000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_melm_wwm_span_noitm_step3w-cm_mask_prompt_lr${lr}_trnval/${cvNo}
#         done
# done

################# Part4: Explore WWM + Span tasks + TaskPretain + Finetune ###################################################
#case1: task-pretrained no itm + www + span + finetune text + wav2vec + visual, fintune on AVL --task finetune
# for lr in 3e-5
# do
#     for seed in 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         num_train_steps=1000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_text --use_visual --use_speech  \
#                 --seed ${seed} \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm-5tasks_wwm_span_noitm_trnval/${cvNo}/ckpt/model_step_2000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-taskpretrain-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm-5tasks_wwm_span_noitm_trnval-lr${lr}_seed${seed}_trnval
#         done
#     done
# done

# ################# Part5: Explore WWM + Span tasks + TaskPretain + Prompt ###################################################
# # case1: all seven cases
# for lr in 3e-5 2e-5 5e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm-5tasks_wwm_span_noitm_trnval/${cvNo}/ckpt/model_step_2000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 4000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_taskpretrain-basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm-cm_mask_prompt_lr${lr}_seed${seed}/${cvNo}
#         done
#     done
# done


# ################# Part6: few-shot of directly train ###################################################
#注意不同比例的数据采用不同的 训练步数 和 valid step, 完整的训练数据5000/32=156steps/epoch, 1500steps. 从头开始训练，所以迭代次数多一些。
#  10%数据 valid step=20, steps=200 * 2
#  20%数据 valid step=40, steps=400 * 2
#  40%数据 valid step=60, steps=600 * 2
#  60%数据 valid step=80, steps=800 * 2
# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         partration=0.1
#         valid_steps=20
#         num_train_steps=400
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --seed ${seed} --cvNo ${cvNo} --use_speech --use_visual --use_text \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_part${partration}.json \
#                 --checkpoint /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-directTrain-uniter3m_visual_wav2vec_text-new-lr${lr}_train${num_train_steps}_trnval_part${partration}_seed${seed}
#         done
#     done
# done

# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         partration=0.2
#         valid_steps=40
#         num_train_steps=800
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --seed ${seed} --cvNo ${cvNo} --use_speech --use_visual --use_text \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_part${partration}.json \
#                 --checkpoint /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-directTrain-uniter3m_visual_wav2vec_text-new-lr${lr}_train${num_train_steps}_trnval_part${partration}_seed${seed}
#         done
#     done
# done

# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         partration=0.4
#         valid_steps=60
#         num_train_steps=1200
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --seed ${seed} --cvNo ${cvNo} --use_speech --use_visual --use_text \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_part${partration}.json \
#                 --checkpoint /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-directTrain-uniter3m_visual_wav2vec_text-new-lr${lr}_train${num_train_steps}_trnval_part${partration}_seed${seed}
#         done
#     done
# done


# for lr in 3e-5
# do
#     for seed in 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         partration=0.6
#         valid_steps=80
#         num_train_steps=1600
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --seed ${seed} --cvNo ${cvNo} --use_speech --use_visual --use_text \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_part${partration}.json \
#                 --checkpoint /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-directTrain-uniter3m_visual_wav2vec_text-new-lr${lr}_train${num_train_steps}_trnval_part${partration}_seed${seed}
#         done
#     done
# done

# ################# Part7: few-shot of finetune ###################################################
#注意不同比例的数据采用不同的 训练步数 和 valid step, 完整的训练数据5000/32=156steps/epoch, 1500steps.
#  10%数据 valid step=20, steps=200 + 100
#  20%数据 valid step=40, steps=400 + 100
#  40%数据 valid step=60, steps=600 + 100
#  60%数据 valid step=80, steps=800 + 100
# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         partration=0.1
#         valid_steps=20
#         num_train_steps=300
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --seed ${seed} --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_part${partration}.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-new-lr${lr}_train${num_train_steps}_trnval_part${partration}_seed${seed}
#         done
#     done
# done

# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         partration=0.2
#         valid_steps=40
#         num_train_steps=500
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --seed ${seed} --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_part${partration}.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-new-lr${lr}_train${num_train_steps}_trnval_part${partration}_seed${seed}
#         done
#     done
# done

# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         partration=0.4
#         valid_steps=60
#         num_train_steps=700
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --seed ${seed} --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_part${partration}.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-new-lr${lr}_train${num_train_steps}_trnval_part${partration}_seed${seed}
#         done
#     done
# done

# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         partration=0.6
#         valid_steps=80
#         num_train_steps=900
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --seed ${seed} --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_part${partration}.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-new-lr${lr}_train${num_train_steps}_trnval_part${partration}_seed${seed}
#         done
#     done
# done

# ################# Part8: few-shot of prompt + seven augment ###################################################
#注意不同比例的数据采用不同的 训练步数 和 valid step, 完整的训练数据5000/32=156steps/epoch, 1500steps. prompt 同样采用direct-train的策略
#  10%数据 valid step=20, steps=200*2
#  20%数据 valid step=40, steps=400*2
#  40%数据 valid step=60, steps=600*2
#  60%数据 valid step=80, steps=800*2

# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         partration=0.1
#         valid_steps=20
#         num_train_steps=400
#         train_batch_size=32
#         inf_batch_size=32
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_part${partration}.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size ${train_batch_size} --val_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --warmup_steps ${valid_steps} --valid_steps ${valid_steps} \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}-basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm-new-cm_mask_prompt_lr${lr}_trnval_part${partration}_seed${seed}/${cvNo}
#         done
#     done
# done

# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         partration=0.2
#         valid_steps=40
#         num_train_steps=800
#         train_batch_size=32
#         inf_batch_size=32
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_part${partration}.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size ${train_batch_size} --val_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --warmup_steps ${valid_steps} --valid_steps ${valid_steps} \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}-basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm-new-cm_mask_prompt_lr${lr}_trnval_part${partration}_seed${seed}/${cvNo}
#         done
#     done
# done

# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         partration=0.4
#         valid_steps=60
#         num_train_steps=1200
#         train_batch_size=32
#         inf_batch_size=32
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_part${partration}.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size ${train_batch_size} --val_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --warmup_steps ${valid_steps} --valid_steps ${valid_steps} \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}-basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm-new-cm_mask_prompt_lr${lr}_trnval_part${partration}_seed${seed}/${cvNo}
#         done
#     done
# done

# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         partration=0.6
#         valid_steps=80
#         num_train_steps=1600
#         train_batch_size=32
#         inf_batch_size=32
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type iam --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_part${partration}.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size ${train_batch_size} --val_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --warmup_steps ${valid_steps} --valid_steps ${valid_steps} \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}-basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm-new-cm_mask_prompt_lr${lr}_trnval_part${partration}_seed${seed}/${cvNo}
#         done
#     done
# done

# ################# Part9: few-shot of prompt + only lav ###################################################
#注意不同比例的数据采用不同的 训练步数 和 valid step, 完整的训练数据5000/32=156steps/epoch, 1500steps. prompt 同样采用direct-train的策略
#  10%数据 valid step=20, steps=200*2
#  20%数据 valid step=40, steps=400*2
#  40%数据 valid step=60, steps=600*2
#  60%数据 valid step=80, steps=800*2
# for lr in 3e-5
# do
#     for seed in 1234 4321 5678 
#     do
#             for cvNo in $(seq 1 10)
#             do
    #         partration=0.1
    #         valid_steps=20
    #         num_train_steps=400
    #         train_batch_size=32
    #         inf_batch_size=32
#             CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                             --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                             --prompt_type iam --seed ${seed} \
#                             --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_onlylva_part${partration}.json \
#                             --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                             --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                             --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                             --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                             --train_batch_size ${train_batch_size} --val_batch_size ${inf_batch_size} \
#                             --num_train_steps ${num_train_steps} --warmup_steps ${valid_steps} --valid_steps ${valid_steps} \
#                             --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}-basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm-cm_mask_prompt_onlylva_lr${lr}_trnval_part${partration}_seed${seed}/${cvNo}
#             done
#         done
# done


# for lr in 3e-5
# do
#     for seed in 1234 4321 5678 
#     do
#             for cvNo in $(seq 1 10)
#             do
    #         partration=0.2
    #         valid_steps=40
    #         num_train_steps=800
    #         train_batch_size=32
    #         inf_batch_size=32
#             CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                             --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                             --prompt_type iam --seed ${seed} \
#                             --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_onlylva_part${partration}.json \
#                             --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                             --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                             --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                             --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                             --train_batch_size ${train_batch_size} --val_batch_size ${inf_batch_size} \
#                             --num_train_steps ${num_train_steps} --warmup_steps ${valid_steps} --valid_steps ${valid_steps} \
#                             --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}-basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm-cm_mask_prompt_onlylva_lr${lr}_trnval_part${partration}_seed${seed}/${cvNo}
#             done
#         done
# done

# for lr in 3e-5
# do
#     for seed in 1234 4321 5678 
#     do
#             for cvNo in $(seq 1 10)
#             do
#             partration=0.4
#             valid_steps=60
#             num_train_steps=1200
#             train_batch_size=32
#             inf_batch_size=32
#             CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                             --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                             --prompt_type iam --seed ${seed} \
#                             --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_onlylva_part${partration}.json \
#                             --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                             --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                             --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                             --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                             --train_batch_size ${train_batch_size} --val_batch_size ${inf_batch_size} \
#                             --num_train_steps ${num_train_steps} --warmup_steps ${valid_steps} --valid_steps ${valid_steps} \
#                             --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}-basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm-cm_mask_prompt_onlylva_lr${lr}_trnval_part${partration}_seed${seed}/${cvNo}
#             done
#         done
# done

# for lr in 3e-5
# do
#     for seed in 5678 
#     do
#             for cvNo in $(seq 10 10)
#             do
#             partration=0.6
#             valid_steps=80
#             num_train_steps=1600
#             train_batch_size=32
#             inf_batch_size=32
#             CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                             --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                             --prompt_type iam --seed ${seed} \
#                             --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_onlylva_part${partration}.json \
#                             --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                             --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                             --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                             --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                             --train_batch_size ${train_batch_size} --val_batch_size ${inf_batch_size} \
#                             --num_train_steps ${num_train_steps} --warmup_steps ${valid_steps} --valid_steps ${valid_steps} \
#                             --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}-basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm-cm_mask_prompt_onlylva_lr${lr}_trnval_part${partration}_seed${seed}/${cvNo}
#             done
#         done
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