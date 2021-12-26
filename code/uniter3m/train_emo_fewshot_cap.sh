
source deactivate base
export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='iemocap'
corpus_name_L='IEMOCAP'

## Outline 
################# Part1: few-shot of directly train ###################################################
################# Part2: few-shot of finetune ###################################################
################# Part3: few-shot of prompt + seven augment ###################################################

# ################# Part1: few-shot of directly train ###################################################
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

# ################# Part2: few-shot of finetune ###################################################
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_mores5.5v5.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.4s.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.5s.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.6s.6_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_onespans.4v.4_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_onespans.5v.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_onespans.6v.6_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 

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
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_lr5e5_bs800/ckpt/model_step_60000.pt  \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_train6w-new-lr${lr}_train${num_train_steps}_trnval_part${partration}_seed${seed}
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
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_lr5e5_bs800/ckpt/model_step_60000.pt  \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_train6w-new-lr${lr}_train${num_train_steps}_trnval_part${partration}_seed${seed}
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
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_lr5e5_bs800/ckpt/model_step_60000.pt  \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_train6w-new-lr${lr}_train${num_train_steps}_trnval_part${partration}_seed${seed}
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
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_lr5e5_bs800/ckpt/model_step_60000.pt  \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_train6w-new-lr${lr}_train${num_train_steps}_trnval_part${partration}_seed${seed}
#         done
#     done
# done

# ################# Part3: few-shot of prompt + seven augment ###################################################
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_mores5.5v5.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.5s.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 
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
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_mores5.5v5.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt 
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.5s.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_lr5e5_bs800/ckpt/model_step_40000.pt
# /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_onespans.5v.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt

#注意不同比例的数据采用不同的 训练步数 和 valid step, 完整的训练数据 5000/32=156steps/epoch, 1500steps. prompt 同样采用 direct-train 的策略
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
#             partration=0.1
#             valid_steps=20
#             num_train_steps=400
#             train_batch_size=32
#             inf_batch_size=32
#             CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                             --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                             --prompt_type iam --seed ${seed} \
#                             --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_onlylva_part${partration}.json \
#                             --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                             --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_lr5e5_bs800/ckpt/model_step_60000.pt \
#                             --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                             --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                             --train_batch_size ${train_batch_size} --val_batch_size ${inf_batch_size} \
#                             --num_train_steps ${num_train_steps} --warmup_steps ${valid_steps} --valid_steps ${valid_steps} \
#                             --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}-basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm-cm_mask_prompt_onlylva_lr${lr}_trnval_part${partration}_seed${seed}/${cvNo}
#             done
#     done
# done

# for lr in 3e-5
# do
#     for seed in 1234 4321 5678 
#     do
#             for cvNo in $(seq 1 10)
#             do
#             partration=0.2
#             valid_steps=40
#             num_train_steps=800
#             train_batch_size=32
#             inf_batch_size=32
#             CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                             --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                             --prompt_type iam --seed ${seed} \
#                             --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_onlylva_part${partration}.json \
#                             --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                             --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.5s.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                             --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                             --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                             --train_batch_size ${train_batch_size} --val_batch_size ${inf_batch_size} \
#                             --num_train_steps ${num_train_steps} --warmup_steps ${valid_steps} --valid_steps ${valid_steps} \
#                             --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}-basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_mmrm_msrm_maskprobv.5s.5_noitm-cm_mask_prompt_onlylva_lr${lr}_trnval_part${partration}_seed${seed}/${cvNo}
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
#                             --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.5s.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                             --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                             --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                             --train_batch_size ${train_batch_size} --val_batch_size ${inf_batch_size} \
#                             --num_train_steps ${num_train_steps} --warmup_steps ${valid_steps} --valid_steps ${valid_steps} \
#                             --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}-basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_mmrm_msrm_maskprobv.5s.5_noitm-cm_mask_prompt_onlylva_lr${lr}_trnval_part${partration}_seed${seed}/${cvNo}
#             done
#         done
# done

# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#             for cvNo in $(seq 1 10)
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
#                             --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.5s.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                             --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                             --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                             --train_batch_size ${train_batch_size} --val_batch_size ${inf_batch_size} \
#                             --num_train_steps ${num_train_steps} --warmup_steps ${valid_steps} --valid_steps ${valid_steps} \
#                             --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}-basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_mmrm_msrm_maskprobv.5s.5_noitm-cm_mask_prompt_onlylva_lr${lr}_trnval_part${partration}_seed${seed}/${cvNo}
#             done
#         done
# done