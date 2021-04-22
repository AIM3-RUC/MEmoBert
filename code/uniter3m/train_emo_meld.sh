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

for norm_type in movienorm selfnorm; 
do
        for lr in 1e-5 2e-5;
        do
                num_train_steps=5000
                valid_steps=500
                train_batch_size=32
                inf_batch_size=32
                frozens=0
                CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
                        --cvNo 1 --model_config config/uniter-base-emoword_nomultitask.json \
                        --cls_num 7 --use_visual --use_speech \
                        --config config/train-emo-${corpus_name}-openface_${norm_type}-base-2gpu.json \
                        --checkpoint /data7/emobert/resources/pretrained/uniter-base-uncased-init.pt \
                        --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
                        --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps ${valid_steps} \
                        --conf_th 0.5 --max_bb 36 --min_bb 10 \
                        --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
                        --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
                        --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
                        --output_dir /data7/emobert/exp/evaluation/MELD/finetune/nomask-direct_bert_base-lr${lr}_train${num_train_steps}_${norm_type}_trnval
        done
done