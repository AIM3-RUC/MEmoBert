# eg:
# bash run.sh V 0 1
export PYTHONPATH=/data9/MEmoConv
set -e
gpu=$1

# 单模态的实验
for modality in L;
do
    for run_idx in 1;
    do
        cmd="python run_baseline.py --gpu_id $gpu --modality=$modality 
            --dataset_mode iemocap_ori
            --pretained_ft_type utt_baseline
            --num_threads 0 --run_idx=$run_idx
            --max_epoch 50 --patience 10 --fix_lr_epoch 20 --warmup_epoch 3
            --dropout_rate 0.5  --learning_rate 3e-4 --batch_size 64 --postfix self
            --v_ft_type affectdenseface --v_input_size 342 --max_visual_tokens 70 
            --a_ft_type wav2vec_zh --a_input_size 1024 --max_acoustic_tokens 150
            --l_ft_type robert_base_wwm_chinese --l_input_size 768 --max_text_tokens 30
            --l_hidden_size 256 --v_hidden_size 256 --a_hidden_size 512 --mid_fusion_layers '256,128'
        "
        echo "\n-------------------------------------------------------------------------------------"
        echo "Execute command: $cmd"
        echo "-------------------------------------------------------------------------------------\n"
        echo $cmd | sh
    done
done

# # 多模态的实验
# for modality in LA LV AV LAV;
# do
#     for run_idx in 1 2 3;
#     do
#         cmd="python run_baseline.py --gpu_id $gpu --modality=$modality 
#             --dataset_mode chmed
#             --pretained_ft_type utt_baseline
#             --num_threads 0 --run_idx=$run_idx
#             --max_epoch 50 --patience 10 --fix_lr_epoch 20 --warmup_epoch 3
#             --dropout_rate 0.5  --learning_rate 3e-4 --batch_size 64 --postfix self
#             --v_ft_type affectdenseface --v_input_size 342 --max_visual_tokens 70 
#             --a_ft_type wav2vec_zh --a_input_size 1024 --max_acoustic_tokens 150
#             --l_ft_type robert_base_wwm_chinese --l_input_size 768 --max_text_tokens 30
#             --l_hidden_size 256 --v_hidden_size 256 --a_hidden_size 512 --mid_fusion_layers '512,256'
#         "
#         echo "\n-------------------------------------------------------------------------------------"
#         echo "Execute command: $cmd"
#         echo "-------------------------------------------------------------------------------------\n"
#         echo $cmd | sh
#     done
# done

# --mid_fusion_layers '256,128'  # 单模态
# --mid_fusion_layers '512,256'  # 多模态