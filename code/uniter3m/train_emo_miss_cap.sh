source deactivate base
export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='iemocap'
corpus_name_L='IEMOCAP'

################# Part0: Direct/BERT train on different modalities and testing on unified model (Data-augmentation) ########################
################# Part1: Freeze the MEmoBert and Only Finetune part parameters ################################################### 
################# Part2: Finetune on different modalities and testing on unified model (Data-augmentation) ########################
################# Part3: Prompt on different modalities and testing on unified model ########################
################# Part4: Flexiable Prompt on different modalities and testing on unified model ########################
################# Part5: Explore ComparE speech features ################################################### 

################# Part0: Direct/BERT train on different modalities and testing on unified model (Data-augmentation) ########################
# for lr in 5e-5 1e-4
# do
#     for seed in 42 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         num_train_steps=3000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo_miss.py \
#                 --cvNo ${cvNo} --use_speech --use_visual --use_text \
#                 --seed ${seed} \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_miss.json \
#                 --checkpoint /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 100 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/miss-nomask-directTrain-uniter3m_visual_wav2vec_text-lr${lr}_train${num_train_steps}_trnval_seed${seed}
#         done
#     done
# done

# for lr in 5e-5 1e-4
# do
#     for seed in 42 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         num_train_steps=3000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo_miss.py \
#                 --cvNo ${cvNo} --use_speech --use_visual --use_text \
#                 --seed ${seed} \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_miss.json \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/miss-nomask-directTrain-FromScratch-uniter3m_visual_wav2vec_text-lr${lr}_train${num_train_steps}_trnval_seed${seed}
#         done
#     done
# done


################# Part1: Freeze the MEmoBert and Only Finetune part parameters ################################################### 
# for frozens in 12 10 8 4 0
# do
#         for lr in 3e-5 3e-4
#         do
#                 for cvNo in $(seq 1 10)
#                 do
#                 num_train_steps=2000
#                 valid_steps=100
#                 train_batch_size=32
#                 inf_batch_size=32
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                         --cvNo ${cvNo} --use_text \
#                         --seed 1234 \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --corpus_name ${corpus_name} --cls_num 4 \
#                         --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                         --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                         --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/FrozeBackbone${frozens}Layers-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-lr${lr}_onlyT_trnval
#                 done
#         done
# done

# for frozens in 12 10 8 4 0
# do
#         for lr in 3e-5 3e-4
#         do
#                 for cvNo in $(seq 1 10)
#                 do
#                 num_train_steps=2000
#                 valid_steps=100
#                 train_batch_size=32
#                 inf_batch_size=32
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                         --cvNo ${cvNo} --use_visual \
#                         --seed 1234 \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --corpus_name ${corpus_name} --cls_num 4 \
#                         --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                         --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                         --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/FrozeBackbone${frozens}Layers-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-lr${lr}_onlyV_trnval
#                 done
#         done
# done


# for frozens in 12 10 8 4 0
# do
#         for lr in 3e-5 3e-4
#         do
#                 for cvNo in $(seq 1 10)
#                 do
#                 num_train_steps=2000
#                 valid_steps=100
#                 train_batch_size=32
#                 inf_batch_size=32
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                         --cvNo ${cvNo} --use_speech \
#                         --seed 1234 \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --corpus_name ${corpus_name} --cls_num 4 \
#                         --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                         --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                         --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/FrozeBackbone${frozens}Layers-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-lr${lr}_onlyS_trnval
#                 done
#         done
# done

# for frozens in 12 10 8 4 0
# do
#         for lr in 3e-5 3e-4
#         do
#                 for cvNo in $(seq 1 10)
#                 do
#                 num_train_steps=2000
#                 valid_steps=100
#                 train_batch_size=32
#                 inf_batch_size=32
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                         --cvNo ${cvNo} --use_text --use_speech \
#                         --seed 1234 \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --corpus_name ${corpus_name} --cls_num 4 \
#                         --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                         --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                         --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/FrozeBackbone${frozens}Layers-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-lr${lr}_onlyTS_trnval
#                 done
#         done
# done


# for frozens in 12 10 8 4 0
# do
#         for lr in 3e-5 3e-4
#         do
#                 for cvNo in $(seq 1 10)
#                 do
#                 num_train_steps=2000
#                 valid_steps=100
#                 train_batch_size=32
#                 inf_batch_size=32
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                         --cvNo ${cvNo} --use_text --use_visual \
#                         --seed 1234 \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --corpus_name ${corpus_name} --cls_num 4 \
#                         --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                         --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                         --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/FrozeBackbone${frozens}Layers-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-lr${lr}_onlyTV_trnval
#                 done
#         done
# done


# for frozens in 12 10 8 4 0
# do
#         for lr in 3e-5 3e-4
#         do
#                 for cvNo in $(seq 1 10)
#                 do
#                 num_train_steps=2000
#                 valid_steps=100
#                 train_batch_size=32
#                 inf_batch_size=32
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                         --cvNo ${cvNo} --use_speech --use_visual \
#                         --seed 1234 \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --corpus_name ${corpus_name} --cls_num 4 \
#                         --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                         --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                         --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/FrozeBackbone${frozens}Layers-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-lr${lr}_onlyVS_trnval
#                 done
#         done
# done


# for frozens in 12 10 8 4 0
# do
#         for lr in 3e-5 3e-4
#         do
#                 for cvNo in $(seq 1 10)
#                 do
#                 num_train_steps=2000
#                 valid_steps=100
#                 train_batch_size=32
#                 inf_batch_size=32
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                         --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                         --seed 1234 \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --corpus_name ${corpus_name} --cls_num 4 \
#                         --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                         --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                         --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/FrozeBackbone${frozens}Layers-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-lr${lr}_onlyTVS_trnval
#                 done
#         done
# done

# for frozens in 4
# do
#         for lr in 3e-5 3e-4
#         do
#                 for cvNo in $(seq 6 10)
#                 do
#                 num_train_steps=2000
#                 valid_steps=100
#                 train_batch_size=32
#                 inf_batch_size=32
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                         --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                         --seed 1234 \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --corpus_name ${corpus_name} --cls_num 4 \
#                         --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                         --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                         --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/FrozeBackbone${frozens}Layers-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w-lr${lr}_onlyTVS_trnval
#                 done
#         done
# done

################# Part2: Finetune on different modalities and testing on unified model (Data-augmentation) ########################
# # 7cases * (5000/32) * 8 = 8000
# nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.5s.5_noitm_lr5e5_bs800
# nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.3s.3_noitm_lr5e5_bs800
# nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_mores5.5v5.5_noitm_lr5e5_bs800
# nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_7cases_noitm_lr5e5_bs800
# nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_7cases_noitm_mores5.5v5.5_lr5e5_bs800/
# nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_lr5e5_bs800/ckpt/model_step_60000.pt
# nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1t.1_noitm_lr5e5_bs800/ckpt/model_step_60000.pt
# 学习率的问题，参数少需要大学习率，全部Finetune的结果也都很差, 因此这里还是采用正常的学习率
# for seed in 4321
# do
#     for frozens in 0
#     do
#         for lr in 3e-5 5e-5
#         do
#             for cvNo in $(seq 1 10)
#             do
#             num_train_steps=8000
#             valid_steps=200
#             train_batch_size=32
#             inf_batch_size=32
#             CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo_miss.py \
#                     --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                     --seed ${seed} \
#                     --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                     --corpus_name ${corpus_name} --cls_num 4 \
#                     --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_miss.json \
#                     --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                     --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                     --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                     --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                     --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                     --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                     --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/miss-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w_froze${frozens}-lr${lr}_trnval_trn${num_train_steps}_seed${seed}
#             done
#         done
#     done
# done

# for seed in 1234 4321 5678
# do
#     for frozens in 0
#     do
#         for lr in 5e-5
#         do 
#             for cvNo in $(seq 1 10)
#             do
#             num_train_steps=2000
#             valid_steps=100
#             train_batch_size=32
#             inf_batch_size=32
#             CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo_miss.py \
#                     --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                     --seed ${seed} \
#                     --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                     --corpus_name ${corpus_name} --cls_num 4 \
#                     --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_miss.json \
#                     --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_mores5.5v5.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                     --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                     --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                     --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                     --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                     --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                     --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/miss-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_mores5.5v5.5_noitm_train4w_froze${frozens}-lr${lr}_trnval_seed${seed}
#             done
#         done
#     done
# done


# for seed in 1234 4321 5678
# do
#     for frozens in 0
#     do
#         for lr in 5e-5
#         do 
#             for cvNo in $(seq 1 10)
#             do
#             num_train_steps=2000
#             valid_steps=100
#             train_batch_size=32
#             inf_batch_size=32
#             CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo_miss.py \
#                     --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                     --seed ${seed} \
#                     --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                     --corpus_name ${corpus_name} --cls_num 4 \
#                     --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_miss.json \
#                     --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.3s.3_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                     --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                     --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                     --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                     --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                     --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                     --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/miss-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.3s.3_noitm_train4w_froze${frozens}-lr${lr}_trnval_seed${seed}
#             done
#         done
#     done
# done

# for seed in 1234 4321 5678
# do
#     for frozens in 0
#     do
#         for lr in 5e-5
#         do 
#             for cvNo in $(seq 1 10)
#             do
#             num_train_steps=2000
#             valid_steps=100
#             train_batch_size=32
#             inf_batch_size=32
#             CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo_miss.py \
#                     --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                     --seed ${seed} \
#                     --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                     --corpus_name ${corpus_name} --cls_num 4 \
#                     --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_miss.json \
#                     --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.5s.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                     --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                     --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                     --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                     --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                     --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                     --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/miss-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.5s.5_noitm_train4w_froze${frozens}-lr${lr}_trnval_seed${seed}
#             done
#         done
#     done
# done

# for seed in 4321 5678
# do
#     for frozens in 0
#     do
#         for lr in 3e-5
#         do 
#             for cvNo in $(seq 1 10)
#             do
#             num_train_steps=2000
#             valid_steps=100
#             train_batch_size=32
#             inf_batch_size=32
#             CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo_miss.py \
#                     --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                     --seed ${seed} \
#                     --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                     --corpus_name ${corpus_name} --cls_num 4 \
#                     --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_miss.json \
#                     --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_7cases_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                     --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                     --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                     --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                     --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                     --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                     --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/miss-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_7cases_noitm_train4w_froze${frozens}-lr${lr}_trnval_seed${seed}
#             done
#         done
#     done
# done

# for seed in 4321 5678
# do
#     for frozens in 0
#     do
#         for lr in 3e-5
#         do 
#             for cvNo in $(seq 1 10)
#             do
#             num_train_steps=2000
#             valid_steps=100
#             train_batch_size=32
#             inf_batch_size=32
#             CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo_miss.py \
#                     --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                     --seed ${seed} \
#                     --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                     --corpus_name ${corpus_name} --cls_num 4 \
#                     --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_miss.json \
#                     --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_7cases_noitm_mores5.5v5.5_lr5e5_bs800/ckpt/model_step_40000.pt \
#                     --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                     --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                     --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                     --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                     --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                     --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/miss-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_7cases_noitm_mores5.5v5.5_noitm_train4w_froze${frozens}-lr${lr}_trnval_seed${seed}
#             done
#         done
#     done
# done

### wwm_mrm_msrm_maskprobv.1s.1, whole modality masking
# for seed in 42 1234 4321 5678
# do
#     for frozens in 0
#     do
#         for lr in 3e-5 5e-5
#         do 
#             for cvNo in $(seq 1 10)
#             do
#             num_train_steps=2000
#             valid_steps=100
#             train_batch_size=32
#             inf_batch_size=32
#             CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo_miss.py \
#                     --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                     --seed ${seed} \
#                     --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                     --corpus_name ${corpus_name} --cls_num 4 \
#                     --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_miss.json \
#                     --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_lr5e5_bs800/ckpt/model_step_60000.pt \
#                     --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                     --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                     --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                     --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                     --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                     --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/miss-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_wwm_mrm_msrm_maskprobv.1s.1_noitm_train6w_froze${frozens}-lr${lr}_trnval_seed${seed}
#             done
#         done
#     done
# done

# ### wwm_mrm_msrm_maskprobv.1s.1t.1, whole modality masking
# for seed in 42 1234 4321 5678
# do
#     for frozens in 0
#     do
#         for lr in 3e-5 5e-5
#         do 
#             for cvNo in $(seq 1 10)
#             do
#             num_train_steps=2000
#             valid_steps=100
#             train_batch_size=32
#             inf_batch_size=32
#             CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo_miss.py \
#                     --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                     --seed ${seed} \
#                     --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                     --corpus_name ${corpus_name} --cls_num 4 \
#                     --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_miss.json \
#                     --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1t.1_noitm_lr5e5_bs800/ckpt/model_step_60000.pt \
#                     --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                     --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                     --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                     --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                     --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                     --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/miss-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_wwm_mrm_msrm_maskprobv.1s.1t.1_noitm_train6w_froze${frozens}-lr${lr}_trnval_seed${seed}
#             done
#         done
#     done
# done

################# Part3: Prompt on different modalities and testing on unified model ########################
# nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800
# nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_mores5.5v5.5_noitm_lr5e5_bs800
# nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_7cases_noitm_lr5e5_bs800
# nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_7cases_noitm_mores5.5v5.5_lr5e5_bs800/
# nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1_noitm_lr5e5_bs800/ckpt/model_step_60000.pt
# nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1t.1_noitm_lr5e5_bs800/ckpt/model_step_60000.pt

# case1: all seven cases -- default model
# for lr in 3e-5
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
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_7cases_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

# case2: all seven cases -- span5 model
# for lr in 3e-5
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
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_mores5.5v5.5_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_mores5.5v5.5_noitm_step4w-cm_mask_prompt_7cases_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

# case3: all seven cases -- 7 cases pretrained model
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
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_7cases_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_7cases_noitm_step4w-cm_mask_prompt_icassp_7cases_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

# case4: all seven cases -- span5 model + 7 cases pretrained model
# for lr in 3e-5
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
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_7cases_noitm_mores5.5v5.5_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_7cases_mores5.5v5.5_noitm_step4w-cm_mask_prompt_7cases_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

# case5: whole modality masking
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

# case6: whole modality masking
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
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_mrm_msrm_maskprobv.1s.1t.1_noitm_lr5e5_bs800/ckpt/model_step_60000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_mrm_msrm_maskprobv.1s.1t.1_noitm_step6w-cm_mask_prompt_icassp_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

### frozen 不同的层，对比finetune的结果，是不是稍微调整模型的情况下prompt比finetune好
# for seed in 1234
# do
#     for frozens in 8
#     do
#         for lr in 3e-4
#         do
#             for cvNo in $(seq 1 10)
#             do
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                             --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                             --prompt_type iam --seed ${seed} --frozen_en_layers ${frozens} \
#                             --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_icassp_onlyl.json \
#                             --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                             --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                             --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                             --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                             --train_batch_size 32 --val_batch_size 32 \
#                             --num_train_steps 2000 --warmup_steps 100 --valid_steps 100 \
#                             --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_Frozen${frozens}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_icassp_onlyl_lr${lr}_trnval_seed${seed}/${cvNo}
#             done
#         done
#     done
# done

# for seed in 1234
# do
#     for frozens in 12
#     do
#         for lr in 3e-4
#         do
#             for cvNo in $(seq 10 10)
#             do
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                             --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                             --prompt_type iam --seed ${seed} --frozen_en_layers ${frozens} \
#                             --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_icassp_onlylv.json \
#                             --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                             --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                             --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                             --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                             --train_batch_size 32 --val_batch_size 32 \
#                             --num_train_steps 2000 --warmup_steps 100 --valid_steps 100 \
#                             --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_Frozen${frozens}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_icassp_onlylv_lr${lr}_trnval_seed${seed}/${cvNo}
#             done
#         done
#     done
# done

# for seed in 1234
# do
#     for frozens in 12 4
#     do
#         for lr in 3e-4
#         do
#             for cvNo in $(seq 1 10)
#             do
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                             --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                             --prompt_type iam --seed ${seed} --frozen_en_layers ${frozens} \
#                             --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_icassp_onlyv.json \
#                             --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                             --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                             --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                             --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                             --train_batch_size 32 --val_batch_size 32 \
#                             --num_train_steps 2000 --warmup_steps 100 --valid_steps 100 \
#                             --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_Frozen${frozens}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_icassp_onlyv_lr${lr}_trnval_seed${seed}/${cvNo}
#             done
#         done
#     done
# done

# for seed in 1234
# do
#     for frozens in 12 8 4
#     do
#         for lr in 3e-4
#         do
#             for cvNo in $(seq 1 10)
#             do
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                             --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                             --prompt_type iam --seed ${seed} --frozen_en_layers ${frozens} \
#                             --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_icassp_onlya.json \
#                             --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                             --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                             --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                             --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                             --train_batch_size 32 --val_batch_size 32 \
#                             --num_train_steps 2000 --warmup_steps 100 --valid_steps 100 \
#                             --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_Frozen${frozens}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_icassp_onlya_lr${lr}_trnval_seed${seed}/${cvNo}
#             done
#         done
#     done
# done

# for seed in 1234
# do
#     for frozens in 12 8 4
#     do
#         for lr in 3e-4
#         do
#             for cvNo in $(seq 1 10)
#             do
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                             --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                             --prompt_type iam --seed ${seed} --frozen_en_layers ${frozens} \
#                             --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_icassp_onlyva.json \
#                             --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                             --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                             --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                             --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                             --train_batch_size 32 --val_batch_size 32 \
#                             --num_train_steps 2000 --warmup_steps 100 --valid_steps 100 \
#                             --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_Frozen${frozens}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_icassp_onlyva_lr${lr}_trnval_seed${seed}/${cvNo}
#             done
#         done
#     done
# done

# for seed in 1234
# do
#     for frozens in 4
#     do
#         for lr in 3e-4 3e-5
#         do
#             for cvNo in $(seq 1 10)
#             do
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                             --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                             --prompt_type iam --seed ${seed} --frozen_en_layers ${frozens} \
#                             --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_prompt_icassp_onlylva.json \
#                             --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                             --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                             --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                             --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                             --train_batch_size 32 --val_batch_size 32 \
#                             --num_train_steps 2000 --warmup_steps 100 --valid_steps 100 \
#                             --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_Frozen${frozens}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_icassp_onlylva_lr${lr}_trnval_seed${seed}/${cvNo}
#             done
#         done
#     done
# done


################# Part4: Flexiable Prompt on different modalities and testing on unified model ########################
# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         prompt_type=iam
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type ${prompt_type} --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_flexprompt.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_flexprompt${prompt_type}_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done

# for lr in 3e-5
# do
#     for seed in 1234 4321 5678
#     do
#         for cvNo in $(seq 1 10)
#         do
#         prompt_type=ifeel
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type ${prompt_type} --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_flexprompt.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size 32 --val_batch_size 32 \
#                         --num_train_steps 3000 --warmup_steps 100 --valid_steps 100 \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_flexprompt${prompt_type}_lr${lr}_trnval_seed${seed}/${cvNo}
#         done
#     done
# done



################# Part5: Explore ComparE speech features ################################################### 
# for seed in 5678
# do
#     for frozens in 0
#     do
#         for lr in 5e-5
#         do
#             for cvNo in $(seq 7 10)
#             do
#             num_train_steps=2000
#             valid_steps=100
#             train_batch_size=32
#             inf_batch_size=32
#             CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo_miss.py \
#                     --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                     --seed ${seed} \
#                     --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                     --corpus_name ${corpus_name} --cls_num 4 \
#                     --config config/downstream/train-emo-${corpus_name}-openface_comparE_selfnorm-base-2gpu-emo_sentiword_miss.json \
#                     --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechcomparE_4tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                     --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                     --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                     --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 130 \
#                     --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                     --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                     --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/miss-nomask-movies-v1v2v3-base-speechcomparE_4tasks_wwm_span_noitm_train4w_froze${frozens}-lr${lr}_trnval_seed${seed}
#             done
#         done
#     done
# done

################# Part5: Flexiable Prompt with task names on different modalities and testing on unified model ########################
# 7cases * (5000/32) * 8 = 8000
# for lr in 3e-5
# do
#     for seed in 4321
#     do
#         for cvNo in $(seq 1 10)
#         do
#         prompt_type=iam
#         num_train_steps=8000
#         valid_steps=200
#         train_batch_size=32
#         inf_batch_size=32
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python pretrain.py \
#                         --cvNo ${cvNo} --n_workers 1 --use_speech --use_visual \
#                         --prompt_type ${prompt_type} --seed ${seed} \
#                         --config config/downstream/pretrain-task-${corpus_name}-base-2gpu_cm_mask_flexpromptmiss_mask.json \
#                         --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
#                         --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
#                         --learning_rate ${lr} --lr_sched_type 'linear'  --gradient_accumulation_steps 1 \
#                         --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
#                         --train_batch_size ${train_batch_size} --val_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --warmup_steps 100 --valid_steps ${valid_steps} \
#                         --output_dir /data7/emobert/exp/prompt_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_flexpromptmiss_mask${prompt_type}_lr${lr}_trnval_trn${num_train_steps}_seed${seed}/${cvNo}
#         done
#     done
# done