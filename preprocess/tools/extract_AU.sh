#!/usr/bin/bash
set -e
input_video=$1
output_dir=$2
echo "OpenFace:FeatureExtraction -f $input_video -aus -out_dir $output_dir"
/root/tools/openface_tool/OpenFace/build/bin/FeatureExtraction -f $input_video -aus -out_dir $output_dir -f

# Usage
# bash extract_AU.sh xxx.mkv output_dir

# example
# bash extract_AU.sh ../resources/output.mkv output