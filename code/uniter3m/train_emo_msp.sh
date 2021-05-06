export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
dropout=0.1
corpus_name='msp'

# case1: text + wav2vec - finetune
for lr in 1e-5 2e-5 5e-5
do
        for cvNo in $(seq 1 12)
        do
        num_train_steps=1000
        valid_steps=100
        train_batch_size=32
        inf_batch_size=32
        frozens=0
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
                --cvNo ${cvNo} --use_speech \
                --model_config config/uniter-base-emoword_nomultitask.json \
                --corpus_name ${corpus_name} --cls_num 4 \
                --config config/train-emo-${corpus_name}-openface_wav2vec-base-2gpu-emo_sentiword.json \
                --checkpoint /data7/emobert/exp/pretrain/nomask_movies_v1v2v3_uniter3m_wav2vec_text_3tasks_mlmitmmsrfr_lr5e5_bs512_faceth0.5/ckpt/model_step_20000.pt \
                --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
                --learning_rate ${lr} --lr_sched_type 'linear' --warmup_steps 0 --patience 5  \
                --IMG_DIM 342 --Speech_DIM 768 \
                --conf_th 0.0 --max_bb 36 --min_bb 10 \
                --speech_conf_th 1.0 --max_frames 64 --min_frames 10 \
                --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
                --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
                --output_dir /data7/emobert/exp/evaluation/MSP/finetune/nomask-movies_v1v2v3_uniter3m_wav2vec_text_3tasks_mlmitmmsrfr-lr${lr}_train${num_train_steps}_trnval
        done
done