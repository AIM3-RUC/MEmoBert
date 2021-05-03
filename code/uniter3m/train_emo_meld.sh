export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='meld'

# for norm_type in movienorm; 
# do
#         for lr in 1e-5 2e-5;
#         do
#                 num_train_steps=2000
#                 valid_steps=200
#                 train_batch_size=32
#                 inf_batch_size=32
#                 frozens=0
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                         --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
#                         --cls_num 7 --use_visual --use_speech \
#                         --config config/train-emo-${corpus_name}-openface_${norm_type}-base-2gpu.json \
#                         --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_4tasks_lr5e5_bs1024_faceth0.5/ckpt/model_step_20000.pt \
#                         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                         --learning_rate ${lr} --lr_sched_type 'linear' \
#                         --conf_th 0.5 --max_bb 36 --min_bb 10 \
#                         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                         --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                         --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask_movies_v1v2v3_uniter3m_4tasks_lr5e5_bs1024_faceth0.5-lr${lr}_th0.5_train${num_train_steps}_${norm_type}_trnval
#         done
# done

# for norm_type in movienorm selfnorm; 
# do
#         for lr in 1e-5 2e-5;
#         do
#                 num_train_steps=5000
#                 valid_steps=500
#                 train_batch_size=32
#                 inf_batch_size=32
#                 frozens=0
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                         --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
#                         --cls_num 7 --use_visual --use_speech \
#                         --config config/train-emo-${corpus_name}-openface_${norm_type}-base-2gpu.json \
#                         --checkpoint /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#                         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                         --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps ${valid_steps} \
#                         --conf_th 0.5 --max_bb 36 --min_bb 10 \
#                         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                         --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                         --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-direct_bert_base-lr${lr}_train${num_train_steps}_${norm_type}_trnval
#         done
# done

# for norm_type in selfnorm; 
# do
#         for lr in 1e-5;
#         do
#                 num_train_steps=2000
#                 valid_steps=200
#                 train_batch_size=32
#                 inf_batch_size=32
#                 frozens=0
#                 CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                         --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
#                         --cls_num 7 --use_visual --use_speech \
#                         --config config/train-emo-${corpus_name}-openface_${norm_type}-base-2gpu.json \
#                         --checkpoint /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-nomask_movies_v1v2v3_uniter3m_4tasks_lr5e5_bs1024-4tasks_maxbb36_faceth0.5_trnval/1/ckpt/model_step_2500.pt \
#                         --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                         --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps ${valid_steps} \
#                         --conf_th 0.5 --max_bb 36 --min_bb 10 \
#                         --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                         --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                         --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                         --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-taskpretrain-movies_v1v2v3_uniter3m_4tasks_lr5e5_bs1024-4tasks_maxbb36_faceth0.5_trnval-lr${lr}_train${num_train_steps}_${norm_type}_trnval
#         done
# done

# for lr in 1e-5 2e-5 5e-5;
# do
#         num_train_steps=3000
#         valid_steps=300
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --corpus_name ${corpus_name} --cls_num 7 --use_speech \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu.json \
#                 --checkpoint   /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 300 \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-directtrain-uniter3m_speech_wav2vec-lr${lr}_train${num_train_steps}_trnval
# done

# for lr in 1e-5 2e-5 5e-5;
# do
#         num_train_steps=3000
#         valid_steps=300
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --corpus_name ${corpus_name} --cls_num 7 --use_speech \
#                 --config config/train-emo-${corpus_name}-openface_wav2vecasr-base-2gpu.json \
#                 --checkpoint   /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 300 \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-directtrain-uniter3m_speech_wav2vecasr-lr${lr}_train${num_train_steps}_trnval
# done

# for lr in 2e-5;
# do
#         num_train_steps=2000
#         valid_steps=200
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
#                 --cls_num 7 --use_visual --use_speech \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu.json \
#                 --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_5tasks_vstype1_lr5e5_bs1024_faceth0.5/ckpt/model_step_20000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps ${valid_steps} --patience 5 \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.5 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-taskpretrain-movies_v1v2v3_uniter3m_wav2vec_5tasks-lr${lr}_train${num_train_steps}_trnval
# done

# for lr in 5e-6
# do
#         for cvNo in $(seq 1 1)
#         do
#         num_train_steps=1500
#         valid_steps=100
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo} --model_config config/uniter-base-emoword_nomultitask.json \
#                 --corpus_name ${corpus_name} --cls_num 7 --use_speech \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu.json \
#                 --checkpoint  /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_speech_wav2vec-berttype-4tasks_maxbb36_faceth0.5_trnval/1/ckpt/model_step_2500.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-taskpretrain-movies_v1v2v3_uniter3m_speech_wav2vec-berttype_text-lr${lr}_train${num_train_steps}_trnval
#         done
# done

# for lr in 1e-5 2e-5 5e-5
# do
#         num_train_steps=2000
#         valid_steps=200
#         train_batch_size=32
#         inf_batch_size=32
#         frozens=0
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo 1 --model_config config/uniter-base-emoword_multitask.json \
#                 --corpus_name ${corpus_name} --cls_num 7 --use_visual \
#                 --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo.json \
#                 --checkpoint  /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_visual_text_4tasks_lr5e5_faceth0.5_melm.5/ckpt/model_step_20000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
#                 --IMG_DIM 342 --Speech_DIM 768 \
#                 --conf_th 0.5 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-movies_v1v2v3_uniter3m_4tasks_melm-lr${lr}_train${num_train_steps}_trnval
# done

for lr in 1e-5 2e-5 5e-5
do
        num_train_steps=1500
        valid_steps=150
        train_batch_size=32
        inf_batch_size=32
        frozens=0
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
                --cvNo 1 --model_config config/uniter-base-emoword_multitask.json \
                --corpus_name ${corpus_name} --cls_num 7 --use_visual \
                --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo.json \
                --checkpoint /data7/emobert/exp/task_pretrain/${corpus_name}_basedon-movies_v1v2v3_uniter3m_visual_text_4tasks_lr5e5_melm.5-faceth0.0_trnval/1/ckpt/model_step_2500.pt \
                --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
                --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
                --IMG_DIM 342 --Speech_DIM 768 \
                --conf_th 0.5 --max_bb 36 --min_bb 10 \
                --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
                --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
                --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
                --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-taskpretrain-movies_v1v2v3_uniter3m_4tasks_melm-lr${lr}_train${num_train_steps}_trnval
done