#!/bin/bash
openface_root="/root/tools/openface_tool/OpenFace/build/bin/"
frame_dir_name="frame"
face_dir_name="openface"
utt_id_lst="utt_ids.txt"


while read utt_id; do
    echo $utt_id
    session_id=${utt_id:4:1}  
    frame_dir=Session"$session_id"/$frame_dir_name/$utt_id
    face_dir=Session"$session_id"/$face_dir_name/$utt_id
    if [ ! -d $face_dir ];then
        mkdir -p $face_dir
    fi
    $openface_root/FaceLandmarkVidMulti -nomask -fdir $frame_dir -out_dir $face_dir/ > /dev/null 2>&1
done < $utt_id_lst
