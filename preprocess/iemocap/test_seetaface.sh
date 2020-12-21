#!/bin/bash

utt_id=$1
test_src="test_src.lst"
test_tgt="test_tgt.lst"
test_dir="test"
session_id=${utt_id:4:1}
dialog_id=`echo $utt_id | cut -d'_' -f1-2`
transcript_file="/data3/lrc/IEMOCAP_full_release/Session$session_id/dialog/transcriptions/${dialog_id}.txt"
echo "RAW transcript info:"
echo "-------------------------------"
cat $transcript_file | grep $utt_id
echo "-------------------------------"

frames=`ls Session${session_id}/frame/$utt_id`
frame_num=`ls Session${session_id}/frame/$utt_id | wc -l`
echo "Frames found: $frame_num"

ls Session${session_id}/frame/$utt_id | awk '{print "'"$pwd/Session${session_id}/frame/$utt_id/"'"$0}' > $test_src
if [ -f $test_tgt ]; then 
    rm $test_tgt
fi
for i in `seq 1 1 $frame_num`;do
    echo $test/$i.jpg >> $test_tgt
done

export LD_LIBRARY_PATH=/data3/lrc/IEMOCAP_full_release/code:/root/anaconda2/pkgs/libopencv-3.4.2-hb342d67_1/lib/libopencv_core.so.3.4:$LD_LIBRARY_PATH
/data2/zjm/tools/seetface/seetaface_detection  /data2/zjm/tools/seetface/seeta_fd_frontal_v1.0.bin  \
    $test_src $test_tgt \
    0 635454

rm $test_src
rm $test_tgt