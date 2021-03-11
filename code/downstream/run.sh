# eg:
# bash run.sh V 0 1
export PYTHONPATH=/data7/MEmoBert
set -e
modality=$1
gpu=$2
run_ind=$3
for run_idx in $run_ind;
do
    for i in `seq 1 10`;
    do
        cmd="python run_baseline.py --gpu_id $gpu --modality=$modality 
            --pretained_ft_type openface_iemocap_raw_img112
            --num_threads 0 --cvNo=$i --run_idx=$run_idx
            --dropout_rate 0.5 --postfix self
            --max_lexical_tokens 22
            --l_hidden_size 128 --v_hidden_size 128 --mid_fusion_layers '256,128'
        "
        # cmd="python run_baseline.py --gpu_id $gpu --modality=$modality 
        #     --pretained_ft_type openface_iemocap_raw_img112
        #     --num_threads 0 --cvNo=$i --run_idx=$run_idx
        #     --dropout_rate 0.5 --postfix www
        #     --max_lexical_tokens 22
        #     --l_hidden_size 128 --v_hidden_size 128 --mid_fusion_layers '256,128'
        # "
        echo "\n-------------------------------------------------------------------------------------"
        echo "Execute command: $cmd"
        echo "-------------------------------------------------------------------------------------\n"
        echo $cmd | sh
    done
done