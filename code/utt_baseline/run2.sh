# eg:
# bash run.sh V 0 1
source deactivate base
export PYTHONPATH=/data9/MEmoConv
set -e
gpu=$1

# 单模态的实验 == pretrained features all 768 dim
for modality in V
do
    for run_idx in 1
    do
        for cvNo in $(seq 3 3)
        do
            python run_baseline.py --gpu_id $gpu --modality=$modality \
                --dataset_mode iemocap_pretrained --cvNo ${cvNo} \
                --pretained_ft_type utt_baseline \
                --num_threads 0 --run_idx=$run_idx  \
                --max_epoch 40 --patience 5 --fix_lr_epoch 20 --warmup_epoch 3 \
                --dropout_rate 0.5  --learning_rate 3e-4 --batch_size 30 --postfix self \
                --v_ft_type '' --v_input_size 768 --max_visual_tokens 50  \
                --a_ft_type '' --a_input_size 768 --max_acoustic_tokens 150 \
                --l_ft_type nomask_movies_v1v2v3_uniter3m_speechwav2vec_5tasks_wwm_span_noitm_lr5e5_bs800_nofinetune  --l_input_size 768 --max_text_tokens 50 \
                --l_hidden_size 256 --v_hidden_size 256 --a_hidden_size 512 --mid_fusion_layers '256,128'
        done
    done
done

# # 多模态的实验
# for modality in LA LV AV LAV;
# do
#     for run_idx in 1 2 3;
#     do
#         python run_baseline.py --gpu_id $gpu --modality=$modality  
#             --dataset_mode iemocap_ori --cvNo ${cvNo} \
#             --pretained_ft_type utt_baseline \
#             --num_threads 0 --run_idx=$run_idx  \
#             --max_epoch 50 --patience 10 --fix_lr_epoch 20 --warmup_epoch 3
#             --dropout_rate 0.5  --learning_rate 3e-4 --batch_size 64 --postfix self
#             --v_ft_type denseface_openface_iemocap_mean_std_torch --v_input_size 342 --max_visual_tokens 70  \
#             --a_ft_type wav2vec_raw --a_input_size 768 --max_acoustic_tokens 150 \
#             --l_ft_type bert --l_input_size 768 --max_text_tokens 50 \
#             --l_hidden_size 256 --v_hidden_size 256 --a_hidden_size 512 --mid_fusion_layers '512,256'
#     done
# done

# --mid_fusion_layers '256,128'  # 单模态
# --mid_fusion_layers '512,256'  # 多模态