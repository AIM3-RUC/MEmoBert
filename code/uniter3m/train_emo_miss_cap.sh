source deactivate base
export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='iemocap'
corpus_name_L='IEMOCAP'

################# Part1: Freeze the MEmoBert and Only Finetune part parameters ################################################### 
################# Part2: Finetune on different modalities and testing on unified model (Data-augmentation) ########################


################# Part1: Freeze the MEmoBert and Only Finetune part parameters ################################################### 
# for frozens in 12 10 8 4 0
# do
#         for lr in 3e-4
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
#         for lr in 3e-4
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
#         for lr in 3e-4
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
#         for lr in 3e-4
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
#         for lr in 3e-4
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
#         for lr in 3e-4
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
#         for lr in 3e-4
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


################# Part2: Finetune on different modalities and testing on unified model (Data-augmentation) ########################
for seed in 1234
do
    for frozens in 0
    do
        for lr in 3e-4
        do
            for cvNo in $(seq 1 1)
            do
            num_train_steps=2000
            valid_steps=100
            train_batch_size=32
            inf_batch_size=32
            CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo_miss.py \
                    --cvNo ${cvNo} --use_text --use_speech --use_visual \
                    --seed 1234 \
                    --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json \
                    --corpus_name ${corpus_name} --cls_num 4 \
                    --config config/downstream/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword_miss.json \
                    --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800/ckpt/model_step_40000.pt \
                    --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
                    --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
                    --max_txt_len 120 --IMG_DIM 342 --Speech_DIM 768 \
                    --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
                    --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
                    --output_dir /data7/emobert/exp/evaluation/${corpus_name_L}/finetune/miss-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w_froze${frozens}-lr${lr}_trnval
            done
        done
    done
done