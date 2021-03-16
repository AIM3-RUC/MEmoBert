#!/usr/bin/bash
# bash extract_features.sh 12 has_active_spk
# 进程数据对应gpus的列表的数目，比如8个进程，gpus=(0 0 1 1 2 2 3 3)，每块卡分两个进程同时跑
split_num=$1  # 12
file_name=$2  # has_active_spk
screen_name=MEmoBert_feature_extra
gpus=(0 1 2 3 4 5 0 1 2 3 4 5)
for i in `seq 0 1 $(($split_num))`; 
    do
    {   
        index=$((i%${#gpus[*]}))
        screen -x -S ${screen_name}_${i} -p 0 -X quit
        screen -dmUS ${screen_name}_${i}
        screen -x -US ${screen_name}_${i} -p 0 -X stuff "source activate torch
        "
        screen -x -US ${screen_name}_${i} -p 0 -X stuff "CUDA_VISIBLE_DEVICES=${gpus[$index]} PYTHONPATH=/data7/MEmoBert/ python extract_features.py $file_name $i $split_num
        "
        # screen -x -US ${screen_name}_${i} -p 0 -X stuff "exit
        # "
        echo "CUDA_VISIBLE_DEVICES=${gpus[$index]} PYTHONPATH=/data7/MEmoBert/ python extract_features.py $file_name $i $split_num"
    }
done

# CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/data7/MEmoBert/ python extract_features.py has_active_spk 0 1