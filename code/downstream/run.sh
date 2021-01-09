# eg:
# bash run.sh AVL 0
export PYTHONPATH=/data7/MEmoBert
set -e
modality=$1
gpu=$2
for run_idx in 4;
do
    for i in `seq 1 1 10`;
    do
        cmd="python run_baseline.py --gpu_id $gpu --modality=$modality 
            --pretained_ft_type denseface_seetaface_iemocap_mean_std
            --num_threads 0 --cvNo=$i --run_idx=$run_idx
            --dropout_rate 0.5 --postfix self
            --l_hidden_size 128 --v_hidden_size 128 --mid_fusion_layers '256,128'
        "
        echo "\n-------------------------------------------------------------------------------------"
        echo "Execute command: $cmd"
        echo "-------------------------------------------------------------------------------------\n"
        echo $cmd | sh
    done
done
# --restore_checkpoint 
# pretained_ft_type = 'denseface_openface_movienomask_mean_std'
# pretained_ft_type = 'denseface_seetaface_movienomask_mean_std'
# pretained_ft_type = 'denseface_seetaface_iemocap_mean_std'