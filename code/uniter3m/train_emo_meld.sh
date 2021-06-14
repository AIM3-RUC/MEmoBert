export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='meld'

# case1: text + wav2vec - finetune
# for lr in 1e-5 2e-5 5e-5
# do
#         num_train_steps=2000
#         valid_steps=200
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo 1 --use_speech \
#                 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --corpus_name ${corpus_name} --cls_num 7 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_wav2vec_text_3tasks_mlmitmmsrfr_lr5e5_bs512_faceth0.5/ckpt/model_step_20000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.5 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-movies_v1v2v3_uniter3m_wav2vec_text_3tasks_mlmitmmsrfr-lr${lr}_train${num_train_steps}_trnval
# done

# ## case1.2: text + wav2vec - pretrain on movies_v1v2v3 + emocls
# for lr in 2e-5 3e-5 5e-5
# do
#         for cvNo in $(seq 1 10)
#         do
#         num_train_steps=1500
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --use_text --use_speech \
#                 --model_config config/uniter-base-emoword_nomultitask_difftype_weaklabelSoft.json\
#                 --corpus_name ${corpus_name} --cls_num 4 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_wav2vec_text_3tasks_emocls_lr5e5_bs512/ckpt/model_step_30000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/IEMOCAP/finetune/nomask_movies_v1v2v3_uniter3m_wav2vec_text_3tasks_emocls-lr${lr}_train${num_train_steps}_trnval
#         done
# done

# case2: text + wav2vec + visual 
# for lr in 1e-5 2e-5 5e-5
# do
#         num_train_steps=2000
#         valid_steps=200
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo 1 --use_speech --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --corpus_name ${corpus_name} --cls_num 7 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype1_lr5e5_bs512_faceth0.5/ckpt/model_step_30000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.5 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_vstype1-lr${lr}_train${num_train_steps}_trnval
# done


# case3: text + wav2vec  - directly train
# for lr in 1e-5 2e-5 5e-5
# do
#         num_train_steps=3000
#         valid_steps=150
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo 1 --use_speech \
#                 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --corpus_name ${corpus_name} --cls_num 7 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-directTrain-uniter3m_wav2vec_text-lr${lr}_train${num_train_steps}_trnval
# done

# case4: text + visual +  wav2vec  - directly train
# for lr in 1e-5 2e-5 5e-5
# do
#         num_train_steps=3000
#         valid_steps=150
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo 1 --use_speech --use_visual \
#                 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --corpus_name ${corpus_name} --cls_num 7 \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#                 --checkpoint /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.5 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-directTrain-uniter3m_visual_wav2vec_text-lr${lr}_train${num_train_steps}_trnval
# done


# case5: text + visual - finetune --pending
# for lr in 1e-5 2e-5 5e-5
# do
#     num_train_steps=2000
#     valid_steps=200
#     train_batch_size=32
#     inf_batch_size=32
#     frozens=0
#     CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#             --cvNo 1 --use_visual \
#             --model_config config/uniter-base-emoword_nomultitask.json \
#             --corpus_name ${corpus_name} --cls_num 7 \
#             --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#             --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_lr5e5_bs512_faceth0.5/ckpt/model_step_20000.pt \
#             --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#             --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#             --IMG_DIM 342 --Speech_DIM 768 \
#             --conf_th 0.5 --max_bb 36 --min_bb 10 \
#             --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#             --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#             --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#             --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-movies_v1v2v3_uniter3m_visual_text_4tasks-lr${lr}_train${num_train_steps}_trnval
# done

# case6: text + visual + sentiment --pending
# for lr in 1e-5 2e-5 5e-5
# do
#     num_train_steps=2000
#     valid_steps=200
#     train_batch_size=32
#     inf_batch_size=32
#     frozens=0
#     CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#             --cvNo 1 --use_visual \
#             --model_config config/uniter-base-emoword_multitask.json \
#             --corpus_name ${corpus_name} --cls_num 7 \
#             --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#             --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_5tasks_vstype1_lr5e5_bs512_faceth0.5_sentiword/ckpt/model_step_20000.pt \
#             --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#             --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#             --IMG_DIM 342 --Speech_DIM 768 \
#             --conf_th 0.5 --max_bb 36 --min_bb 10 \
#             --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#             --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#             --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#             --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-movies_v1v2v3_uniter3m_visual_text_4tasks_melmsentiword-lr${lr}_train${num_train_steps}_trnval
# done

# case7: text + wav2vec + sentiment --pending
# for lr in 1e-5 2e-5 5e-5
# do
#     num_train_steps=2000
#     valid_steps=200
#     train_batch_size=32
#     inf_batch_size=32
#     frozens=0
#     CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#             --cvNo 1 --use_speech \
#             --model_config config/uniter-base-emoword_multitask.json \
#             --corpus_name ${corpus_name} --cls_num 7 \
#             --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
#             --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_wav2vec_text_4tasks_vstype1_lr5e5_bs512_faceth0.5_sentiword/ckpt/model_step_20000.pt \
#             --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#             --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#             --IMG_DIM 342 --Speech_DIM 768 \
#             --conf_th 0.5 --max_bb 36 --min_bb 10 \
#             --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#             --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#             --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#             --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-movies_v1v2v3_uniter3m_wav2vec_text_3tasks_melmsentiword-lr${lr}_train${num_train_steps}_trnval
# done