# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# 语音的处理保持跟
set -e
img_dir=/data7/emobert/norm_comparE_npzs/movies_v1_5mean
output_dir=/data7/emobert/norm_comparE_db/movies_v1_5mean
# img_dir=/data7/emobert/exp/evaluation/IEMOCAP/feature/movies_norm_comparE_npzs/
# output_dir=/data7/emobert/exp/evaluation/IEMOCAP/feature/movies_norm_comparE_db/
# img_dir=/data7/emobert/exp/evaluation/MSP/feature/movies_norm_comparE_npzs/
# output_dir=/data7/emobert/exp/evaluation/MSP/feature/movies_norm_comparE_db/
# img_dir=/data7/emobert/exp/evaluation/MELD/feature/norm_comparE_npzs/
# output_dir=/data7/emobert/exp/evaluation/MELD/feature/norm_comparE_db/
echo "convrting image features ..."
python convert_imgdir.py --img_dir $img_dir --output $output_dir --conf_th 1.0 --max_bb 64 --nproc 16
echo "done"