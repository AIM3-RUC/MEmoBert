export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
frozens=0
dropout=0.1
corpus_name='iemocap'

# case1: directly train based on the text bert-base and speech wav2vec
# for lr in 2e-5 5e-5;
# do
#         for cvNo in `seq 1 10`;
#         do
#         train_batch_size=32
#         inf_batch_size=32
#         num_train_steps=3000
#         valid_steps=100
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo}  --model_config config/uniter-3flow_swav2vec_c4.json \
#                 --cls_num 4 --use_speech \
#                 --config config/train-emo-${corpus_name}-openface_rawwav-base-2gpu.json \
#                 --pretrained_text_checkpoint /data7/emobert/resources/pretrained/bert_base_model.pt \
#                 --pretrained_audio_checkpoint /data7/emobert/resources/pretrained/wav2vec_base/wav2vec_base.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 96000 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/IEMOCAP/finetune/flow3-directTrain-text12-wav2vec2-lr${lr}_th${conf_th}_train${num_train_steps}_trnval
#         done
# done    

# case2: based on stage1 model and finetune
# for lr in 2e-5 5e-5;
# do
#         for cvNo in `seq 1 10`;
#         do
#         train_batch_size=32
#         inf_batch_size=32
#         num_train_steps=3000
#         valid_steps=100
#         CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
#                 --cvNo ${cvNo}  --model_config config/uniter-3flow_swav2vec_c4.json \
#                 --cls_num 4 --use_speech \
#                 --config config/train-emo-${corpus_name}-openface_rawwav-base-2gpu.json \
#                 --checkpoint /data7/emobert/exp/pretrain/flow3-stage1-text12_fix-wav2vec2_fix-cross4Update-nomask_movies_v1v2v3_uniter_mlmitm_lr5e5_notypeemb_train5w/ckpt/model_step_100000.pt \
#                 --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
#                 --learning_rate ${lr} --lr_sched_type 'linear' \
#                 --conf_th 0.0 --max_bb 36 --min_bb 10 \
#                 --speech_conf_th 1.0 --max_frames 96000 --min_frames 10 \
#                 --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
#                 --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
#                 --output_dir /data7/emobert/exp/evaluation/IEMOCAP/finetune/flow3_stage1_text12_fix-wav2vec2_fix-cross4Update_movies_v1v2v3_train10w-lr${lr}_th${conf_th}_train${num_train_steps}_trnval
#         done
# done    


# case2: based on stage12 model and finetune
for lr in 2e-5 5e-5;
do
        for cvNo in `seq 1 10`;
        do
        train_batch_size=32
        inf_batch_size=32
        num_train_steps=3000
        valid_steps=100
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
                --cvNo ${cvNo}  --model_config config/uniter-3flow_swav2vec_c4.json \
                --cls_num 4 --use_speech \
                --config config/train-emo-${corpus_name}-openface_rawwav-base-2gpu.json \
                --checkpoint /data7/emobert/exp/pretrain/flow3-stage12-text12Update-wav2vec2Update-cross4Update-nomask_movies_v1v2v3_uniter_mlmitm_lr5e5_notypeemb_bs800_train10w/ckpt/model_step_45000.pt \
                --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
                --learning_rate ${lr} --lr_sched_type 'linear' \
                --conf_th 0.0 --max_bb 36 --min_bb 10 \
                --speech_conf_th 1.0 --max_frames 96000 --min_frames 10 \
                --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
                --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
                --output_dir /data7/emobert/exp/evaluation/IEMOCAP/finetune/flow3_stage12_text12Update-wav2vec2Update-cross4Update_movies_v1v2v3_train5w-lr${lr}_th${conf_th}_train${num_train_steps}_trnval
        done
done