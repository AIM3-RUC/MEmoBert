# eg:
# bash scripts/train_ef.sh AVL 1 0
set -e
modality=$1
run_idx=$2
gpu=$3
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_10fold --model=early_fusion_multi --gpu_ids=$gpu
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--acoustic_ft_type=IS10 --lexical_ft_type=text --visual_ft_type=denseface
--input_dim_a=1582 --mid_layers_a=512,256,128
--input_dim_v=342 --hidden_size=128 --embd_size_v=128 --embd_method=maxpool
--input_dim_l=1024 --embd_size_l=128
--mid_layers_fusion=256,128 
--output_dim=4 --modality=$modality
--niter=20 --niter_decay=20 --verbose
--batch_size=128 --lr=1e-3 --dropout_rate=0.5 --run_idx=$run_idx
--name=ef --suffix={modality}_run{run_idx}
--cvNo=$i"

# --name=ef_A --suffix=Adnn{mid_layers_a}_Vlstm{hidden_size}_{embd_size_v}_Lcnn{embd_size_l}_fusion{mid_layers_fusion}run{run_idx}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done