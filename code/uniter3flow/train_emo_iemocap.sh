export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
frozens=0
dropout=0.1
corpus_name='iemocap'

for lr in 1e-5 2e-5 5e-5;
do
        for cvNo in `seq 1 10`;
        do
        train_batch_size=28
        inf_batch_size=28
        num_train_steps=1200
        valid_steps=100
        CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
                --cvNo ${cvNo}  --model_config config/uniter-3flow_swav2vec_c4.json \
                --cls_num 4 --use_speech \
                --config config/train-emo-${corpus_name}-openface-base-2gpu.json \
                --checkpoint /data7/emobert/exp/pretrain/flow3-stage1-text12_fix-wav2vec2_fix-cross4Update-nomask_movies_v1v2v3_uniter_mlmitm_lr5e5_notypeemb/ckpt/model_step_20000.pt \
                --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
                --learning_rate ${lr} --lr_sched_type 'linear' --conf_th 0.0 --max_bb ${max_bb} \
                --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
                --output_dir /data7/emobert/exp/evaluation/IEMOCAP/finetune/flow3_text12_fix-wav2vec2_fix_visual4-cross4Update_movies_v1v2v3_uniter-lr${lr}_th${conf_th}_train${num_train_steps}_trnval
        done
done    

