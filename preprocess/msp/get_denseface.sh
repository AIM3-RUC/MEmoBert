#!/usr/bin/bash
set -e
source activate torch
python get_face_pipeline.py

gpus=(4 5 6 7)
source activate py3
total_videos=`ls ../Face/ | wc -l`
for i in `seq 1 1 ${#gpus[*]}`;
do
{
    start=$((total_videos*(i-1)/${#gpus[*]}))
    end=$((total_videos*i/${#gpus[*]}))
    # echo $start $end
    CUDA_VISIBLE_DEVICES=$i python extact_denseface.py $start $end
} &
done 
wait