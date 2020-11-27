#!/usr/bin/bash

split_num=$1
file_name=$2
gpus=(1 2 3 4 5)
for i in `seq 0 1 $(($split_num-1))`; 
    do
    {   
        index=$((i%${#gpus[*]}))
        CUDA_VISIBLE_DEVICES=$gpus[$index] python extract_features.py $file_name $i $split_num
    }
done