#!/bin/bash

frame_dir_name="frame"
face_dir_name="face"
source_list="source.lst"
target_list="target.lst"
utt_id_lst="utt_ids.txt"

if [ -f $source_list ];then
    rm $source_list
fi

if [ -f $target_list ];then 
    rm $target_list
fi

while read utt_id; do
    session_id=${utt_id:4:1}  
    frame_dir=Session"$session_id"/$frame_dir_name/$utt_id
    ls $frame_dir | grep jpg | awk '{print "'"$frame_dir/"'" $0}' >> $source_list
    face_dir=Session"$session_id"/$face_dir_name/$utt_id
    if [ ! -d $face_dir ];then
        mkdir -p $face_dir
    fi
    ls $frame_dir | grep jpg | awk '{print "'"$face_dir/"'" $0}' >> $target_list
done < $utt_id_lst

total_lines=`cat $source_list | wc -l`
echo "$total_lines to be processed"
export LD_LIBRARY_PATH=/data3/lrc/IEMOCAP_full_release/code:/root/anaconda2/pkgs/libopencv-3.4.2-hb342d67_1/lib/libopencv_core.so.3.4:$LD_LIBRARY_PATH
/data2/zjm/tools/seetface/seetaface_detection  /data2/zjm/tools/seetface/seeta_fd_frontal_v1.0.bin  \
                $source_list  \
                $target_list  0 635454
