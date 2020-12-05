#!/usr/bin/bash

split_num=$1
file_name=$2
screen_name=MEmoBert_feature
gpus=(0 0 1 1 2 2 3 3 4 4 5 5)
for i in `seq 0 1 $(($split_num-1))`; 
    do
    {   
        index=$((i%${#gpus[*]}))
        screen -x -S ${screen_name}_${i} -p 0 -X quit
        screen -dmUS ${screen_name}_${i}
        screen -x -US ${screen_name}_${i} -p 0 -X stuff "source activate py3
        "
        screen -x -US ${screen_name}_${i} -p 0 -X stuff "CUDA_VISIBLE_DEVICES=${gpus[$index]} python extract_features.py $file_name $i $split_num
        "
        # screen -x -US ${screen_name}_${i} -p 0 -X stuff "exit
        # "
        echo "CUDA_VISIBLE_DEVICES=${gpus[$index]} python extract_features.py $file_name $i $split_num"
    }
done