#!/bin/bash

# where IEMOCAP_full_release direcory lies.
root="/data3/lrc/IEMOCAP_full_release"

# where to put sentence video
store="sentence_video"

# cut 
prefix="Ses"
for ((i=1;i<=5;i++));do
    {
    if [ ! -d Session"$i"/$store ];then
        mkdir -p Session"$i"/$store
    fi
	for file in `ls $root/Session$i/dialog/transcriptions`;do
	    cat $root/Session$i/dialog/transcriptions/$file | while read line
            do
                # calc ffmpeg cut parameter
                output_name=`echo $line | cut -d" " -f1`
                result=$(echo $output_name | grep ${prefix})
                if [[ "$result" == "" ]];then
                    continue
                fi
                time_info=`echo $line | cut -d" " -f2`
                time_info=`echo $time_info | cut -d"[" -f2 | cut -d"]" -f1`
                st=`echo $time_info | cut -d"-" -f1`
                end=`echo $time_info | cut -d"-" -f2`

                st=`echo $st | awk ' { printf("%5.3f\n", $0); } '`
                end=`echo $end | awk ' { printf("%5.3f\n", $0); } '`

                st_head=`echo $st | cut -d"." -f1`
                st_ta=`echo $st | cut -d"." -f2`
                end_head=`echo $end | cut -d"." -f1`
                duration="0:0:"$(( $end_head-$st_head ))

                st_hour=$(( $st_head/3600 ))
                st_min=$(( ($st_head-${st_hour}*3600)/60 ))
                st_sec=$(( $st_head-${st_hour}*3600-${st_min}*60 ))
                ss=`echo ${st_hour}:${st_min}:${st_sec}.${st_ta}`
                
                # decide use left side or right side
                # In Iemocap dataset the one who where mocap are put in the left
                gender=${output_name:0-4:1}
                mocap=${output_name:5:1}
                if [[ "$gender" == "$mocap" ]];then
                    crop_cmd="crop=iw/2:ih:0:0"
                else
                    crop_cmd="crop=iw/2:ih:iw/2:0"
                fi
                save_path=Session${i}/$store/${output_name}.mp4
                if [ ! -f $save_path ];then
                    echo "Session"$i"/dialog/avi/DivX/"${file%.*}".avi"" ""Session"$i"/sentence_video/"$output_name".mp4"
                    ffmpeg -ss $ss -i $root/Session$i/dialog/avi/DivX/${file%.*}.avi -strict -2  -t $duration -vf $crop_cmd $save_path -y > /dev/null 2>&1
                # else echo "exists $save_path"
                fi
            done
	done
    } &
done