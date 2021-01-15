# eg:
# bash run_pretrained_ft.sh V 0 1
export PYTHONPATH=/data7/MEmoBert
set -e
modality=$1
gpu=$2
run_ind=$3
for run_idx in $run_ind;
do
    for i in `seq 1 1 1`;
    do
        cmd="python run_baseline.py --gpu_id $gpu --modality=$modality 
            --pretained_ft_type nomask_movies_v1_uniter_4tasks_nofinetune
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
# pretained_ft_type = 'nomask_movies_v1_uniter_4tasks_nofinetune/'
# pretained_ft_type = 'nomask_movies_v1_uniter_mlm_itm_2tasks_nofinetune/'
# pretained_ft_type = 'nomask_movies_v1_uniter_mlm_mrckl_2tasks_nofinetune/'
# pretained_ft_type = 'nomask_movies_v1_uniter_mlm_mrfr_2tasks_nofinetune/'
# pretained_ft_type = 'nomask_movies_v1_uniter_mlm_mrfr_mrckl_3tasks_nofinetune/'