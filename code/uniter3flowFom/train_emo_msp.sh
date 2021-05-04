export PYTHONPATH=/data7/MEmoBert
gpu_id=$1
frozens=0
dropout=0.1
corpus_name='msp'

# 关于模型的初始化，同时加载backbone和Uniter, 默认是所有模块全部进行finetune.
for norm_type in selfnorm movienorm; 
do
        for lr in 2e-5 5e-5;
        do
                for cvNo in `seq 1 12`;
                do
                num_train_steps=1000
                valid_steps=100
                train_batch_size=24
                inf_batch_size=24
                CUDA_VISIBLE_DEVICES=${gpu_id} horovodrun -np 1 python train_emo.py \
                        --cvNo ${cvNo} --model_config config/uniter-3flow_s4v4c4.json \
                        --cls_num 4 --use_visual --use_speech \
                        --config config/train-emo-${corpus_name}-openface_${norm_type}-base-2gpu.json \
                        --checkpoint /data7/MEmoBert/emobert/exp/pretrain/flow3_text12_visual4_speech4_cross4_typeEmb-nomask_movies_v1v2v3_uniter_mlmitm_lr5e5-textbackbone_optimFalse-bs480_faceth0.5/ckpt/model_step_20000.pt \
                        --frozen_en_layers ${frozens} --cls_dropout ${dropout} --cls_type vqa --postfix none \
                        --learning_rate ${lr} --lr_sched_type 'linear' \
                        --conf_th 0.0 --max_bb 36 --min_bb 10 \
                        --speech_conf_th 1.0 --max_frames 360 --min_frames 10 \
                        --train_batch_size ${train_batch_size} --inf_batch_size ${inf_batch_size} \
                        --num_train_steps ${num_train_steps} --valid_steps ${valid_steps}  \
                        --output_dir /data7/emobert/exp/evaluation/MSP/finetune/flow3_text12_visual4_speech4_cross4_typeEmb-movies_v1v2v3_uniter_lr5e5-backbone_scratch_optimFalse-lr${lr}_th${conf_th}_train1000_${norm_type}_trnval
                done
        done 
done
