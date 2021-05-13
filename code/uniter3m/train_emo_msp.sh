export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='msp'

# case1: text + wav2vec - finetune on gpu2
# for lr in 1e-5 2e-5 5e-5
# do
#     for frozens in 0 4 6
#     do
#         for cvNo in $(seq 11 12)
#         do
#         num_train_steps=2000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_speech \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_wav2vec_text_3tasks_mlmitmmsrfr_lr5e5_bs1024_train10w/ckpt/model_step_100000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_wav2vec_text_3tasks_mlmitmmsrfr_train10w-lr${lr}_train${num_train_steps}_trnval_frozen${frozens}
#         done
#     done
# done

# # case2: text + wav2vec + visual - on gpu5
# for lr in 1e-5 2e-5 5e-5
# do
#     for frozens in 0 4 6
#     do
#         for cvNo in $(seq 11 12)
#         do
#         num_train_steps=2000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_speech --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2_lr5e5_bs1024_train10w/ckpt/model_step_100000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2_train10w-lr${lr}_train${num_train_steps}_trnval_frozen${frozens}
#         done    
#     done
# done

# case2.1: text + wav2vec + visual - on gpu5
# for lr in 5e-5
# do
#         for cvNo in $(seq 11 12)
#         do
#         num_train_steps=1000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_speech --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2_lr5e5_bs512_faceth0.5/ckpt/model_step_30000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2-lr${lr}_train${num_train_steps}_trnval
#         done
# done

# # # case2.2: text + wav2vec + visual - fintune on AL
# for lr in 2e-5 5e-5
# do
#         for cvNo in $(seq 1 10)
#         do
#         num_train_steps=1500
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_speech \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2_lr5e5_bs512_faceth0.5/ckpt/model_step_30000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2-FinetuneAL-lr${lr}_train${num_train_steps}_trnval
#         done
# done

# # # case2.3: text + wav2vec + visual - fintune on VL
# for lr in 2e-5 5e-5
# do
#         for cvNo in $(seq 1 10)
#         do
#         num_train_steps=1500
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2_lr5e5_bs512_faceth0.5/ckpt/model_step_30000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2-FinetuneVL-lr${lr}_train${num_train_steps}_trnval
#         done
# done

# # # case2.4: text + wav2vec + visual - fintune on L
# for lr in 2e-5 5e-5
# do
#         for cvNo in $(seq 1 10)
#         do
#         num_train_steps=1500
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2_lr5e5_bs512_faceth0.5/ckpt/model_step_30000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2-FinetuneL-lr${lr}_train${num_train_steps}_trnval
#         done
# done

# # case2.5: text + wav2vec + visual - fintune on A
for lr in 2e-5 5e-5
do
        for cvNo in $(seq 1 12)
        do
        num_train_steps=1500
        valid_steps=100
        train_batch_size=32
        inf_batch_size=32
        frozens=0
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
                --cvNo ${cvNo} --use_speech \
                --model_config config/uniter-base-emoword_nomultitask_difftype.json \
                --corpus_name ${corpus_name} --cls_num 4 \
                --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
                --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2_lr5e5_bs512_faceth0.5/ckpt/model_step_30000.pt \
                --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
                --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
                --IMG_DIM 342 --Speech_DIM 768 \
                --conf_th 0.0 --max_bb 36 --min_bb 10 \
                --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
                --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
                --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
                --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2-FinetuneA-lr${lr}_train${num_train_steps}_trnval
        done
done

# # case2.5: text + wav2vec + visual - fintune on A
for lr in 2e-5 5e-5
do
        for cvNo in $(seq 1 12)
        do
        num_train_steps=1500
        valid_steps=100
        train_batch_size=32
        inf_batch_size=32
        frozens=0
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
                --cvNo ${cvNo} --use_visual \
                --model_config config/uniter-base-emoword_nomultitask_difftype.json \
                --corpus_name ${corpus_name} --cls_num 4 \
                --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
                --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2_lr5e5_bs512_faceth0.5/ckpt/model_step_30000.pt \
                --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
                --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
                --IMG_DIM 342 --Speech_DIM 768 \
                --conf_th 0.0 --max_bb 36 --min_bb 10 \
                --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
                --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
                --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
                --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype2-FinetuneV-lr${lr}_train${num_train_steps}_trnval
        done
done

# case3: text + wav2vec  - directly train
# for lr in 1e-5 2e-5 5e-5
# do
#         for cvNo in $(seq 1 12)
#         do
#         num_train_steps=2000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_speech \
#                 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-directTrain-uniter3m_wav2vec_text-lr${lr}_train${num_train_steps}_trnval
#         done
# done

# case4: text + visual +  wav2vec  - directly train
# for lr in 1e-5 2e-5 5e-5
# do
#         for cvNo in $(seq 1 12)
#         do
#         num_train_steps=2000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_speech --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-directTrain-uniter3m_visual_wav2vec_text-lr${lr}_train${num_train_steps}_trnval
#         done
# done

# for lr in 1e-5 2e-5 5e-5
# do
#         for cvNo in $(seq 1 12)
#         do
#         num_train_steps=2000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_speech --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-directTrain-uniter3m_visual_wav2vec_text_vstype2-lr${lr}_train${num_train_steps}_trnval
#         done
# done

## case5: text + visual - finetune --pending
# for lr in 2e-5 5e-5
# do
#         for cvNo in $(seq 1 12)
#         do
#         num_train_steps=1000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_lr5e5_bs512_faceth0.5/ckpt/model_step_20000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_text_4tasks-lr${lr}_train${num_train_steps}_trnval
#         done
# done

# # case6: text + visual/speech + sentiment --pending
# for lr in 2e-5 5e-5
# do
#         for cvNo in $(seq 1 12)
#         do
#         num_train_steps=1000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_visual \
#                 --model_config config/uniter-base-emoword_multitask.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_5tasks_vstype1_lr5e5_bs512_faceth0.5_sentiword/ckpt/model_step_20000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_text_4tasks_melmsentiword-lr${lr}_train${num_train_steps}_trnval
#         # # case7: text + wav2vec + sentiment --pending
#         num_train_steps=1000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_speech \
#                 --model_config config/uniter-base-emoword_multitask.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_wav2vec_text_4tasks_vstype1_lr5e5_bs512_faceth0.5_sentiword/ckpt/model_step_20000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_wav2vec_text_3tasks_melmsentiword-lr${lr}_train${num_train_steps}_trnval
#         done
# done

## case8: text + visual - finetune --pending
# for lr in 1e-5 2e-5 5e-5
# do
#         for cvNo in $(seq 1 10)
#         do
#         num_train_steps=1000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_text_4tasks-faceth0.0_trnval/${cvNo}/ckpt/model_step_1000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-taskpretrain-movies_v1v2v3_uniter3m_visual_text_4tasks-lr${lr}_train${num_train_steps}_trnval
#         done
# done

## case9: taskpretrain + text + speech - finetune --pending
# for lr in 5e-5
# do
#         for cvNo in $(seq 11 12)
#         do
#         num_train_steps=1000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_speech \
#                 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_wav2vec_text_3tasks_lr5e5-faceth0.0_trnval/${cvNo}/ckpt/model_step_1000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-taskpretrain-movies_v1v2v3_uniter3m_wav2vec_text_4tasks-lr${lr}_train${num_train_steps}_trnval
#         done
# done


# case9: text + wav2vec + visual - 6tasks
# for lr in 5e-5
# do
#         for cvNo in $(seq 11 12)
#         do
#         num_train_steps=1000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_speech --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_6tasks_vstype2_lr5e5_bs512_faceth0.5/ckpt/model_step_30000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_wav2vec_text_6tasks_vstype2-lr${lr}_train${num_train_steps}_trnval
#         done
# done

# case10: text + wav2vec + visual - 7tasks
# for lr in 2e-5 5e-5
# do
#     for frozens in 0 4 6
#     do
#         for cvNo in $(seq 1 12)
#         do
#         num_train_steps=2000
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_speech --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_7tasks_vstype2_lr5e5_bs1024_train10w/ckpt/model_step_100000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_wav2vec_text_7tasks_vstype2_train10w-lr${lr}_train${num_train_steps}_trnval_frozen${frozens}
#         done
#     done
# done