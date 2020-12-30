#!/bin/bash

video_dir_name="sentence_video"
save_dir_name="frame"
fps=10

for ((i=1;i<=5;i++));
    do
    {
        video_dir=Session"$i"/$video_dir_name
        for video in `ls $video_dir | grep mp4`;do
            utt_id=`echo $video | cut -d'.' -f1`
            save_dir=Session"$i"/$save_dir_name/$utt_id
            if [ ! -d $save_dir ];then
                mkdir -p $save_dir
                ffmpeg -i $video_dir/$video -f image2  -vf fps=fps=$fps -qscale:v 2 $save_dir/%04d.jpg -y > /dev/null 2>&1
            else
                jpg_num=`ls $save_dir | wc -l`
                if [[ "$jpg_num" == "0" ]];then
                    ffmpeg -i $video_dir/$video -f image2  -vf fps=fps=$fps -qscale:v 2 $save_dir/%04d.jpg -y > /dev/null 2>&1
                fi
            fi
        done
    } &
done
wait
echo "Finished"
