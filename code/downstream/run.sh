# eg:
# bash scripts/train_ef.sh AVL 1 0
export PYTHONPATH=/data7/MEmoBert
set -e
modality=$1
run_idx=$2
gpu=$3 
for i in `seq 1 1 1`;
do
cmd="python run_baseline.py --gpu_id $gpu --modality=$modality 
    --num_threads 0 --cvNo=$i --run_idx=$run_idx
    --dropout_rate 0.5 --lr 1e-4 --postfix none 
    --l_hidden_size 128 --v_hidden_size 128 --mid_fusion_layers '256,128'
   "
echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done

# --restore_checkpoint 