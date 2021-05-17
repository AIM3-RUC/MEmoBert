export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='msp'

# # case2: text + wav2vec + visual - on gpu5
# for lr in 2e-5 5e-5
# do
#         for cvNo in $(seq 1 12)
#         do
#         num_train_steps=1500
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabel.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_4tasks_emoclsSoft_noitm_vstype2_lr5e5_bs1024/ckpt/model_step_30000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_wav2vec_text_4tasks_emoclsSoft_noitm_vstype2_train3w-lr${lr}_train${num_train_steps}_trnval
#     done
# done


# for lr in 2e-5 5e-5
# do
#         for cvNo in $(seq 1 12)
#         do
#         num_train_steps=1500
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_text --use_speech --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabel.json \
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsSoft_withitm_vstype2_lr5e5_bs1024/ckpt/model_step_30000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsSoft_withitm_vstype2_train3w-lr${lr}_train${num_train_steps}_trnval
#     done
# done

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
                --cvNo ${cvNo} --use_text --use_speech --use_visual \
                --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelHard.json \
                --corpus_name ${corpus_name} --cls_num 4 \
                --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
                --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_4tasks_emoclsHard_noitm_vstype2_lr5e5_bs1024/ckpt/model_step_30000.pt \
                --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
                --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
                --IMG_DIM 342 --Speech_DIM 768 \
                --conf_th 0.0 --max_bb 36 --min_bb 10 \
                --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
                --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
                --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
                --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_wav2vec_text_4tasks_emoclsHard_noitm_vstype2_train3w-lr${lr}_train${num_train_steps}_trnval
    done
done

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
                --cvNo ${cvNo} --use_text --use_speech --use_visual \
                --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabel.json \
                --corpus_name ${corpus_name} --cls_num 4 \
                --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
                --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsHard_withitm_vstype2_lr5e5_bs1024/ckpt/model_step_30000.pt \
                --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
                --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
                --IMG_DIM 342 --Speech_DIM 768 \
                --conf_th 0.0 --max_bb 36 --min_bb 10 \
                --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
                --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
                --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
                --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsHard_withitm_vstype2_train3w-lr${lr}_train${num_train_steps}_trnval
    done
done

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
                --cvNo ${cvNo} --use_text --use_speech --use_visual \
                --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabel.json \
                --corpus_name ${corpus_name} --cls_num 4 \
                --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
                --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsLogits_withitm_vstype2_lr5e5_bs1024/ckpt/model_step_30000.pt \
                --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
                --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
                --IMG_DIM 342 --Speech_DIM 768 \
                --conf_th 0.0 --max_bb 36 --min_bb 10 \
                --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
                --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
                --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
                --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_emoclsLogits_withitm_vstype2_train3w-lr${lr}_train${num_train_steps}_trnval
    done
done