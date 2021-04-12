# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# 语音的处理保持跟
set -e
img_dir=/data7/emobert/norm_comparE_npzs/movies_v3
output_dir=/data7/emobert/norm_comparE_db/movies_v3
# img_dir=/data7/emobert/exp/evaluation/IEMOCAP/feature/denseface_openface_iemocap_mean_std_torch/ft_npzs/fc
# output_dir=/data7/emobert/exp/evaluation/IEMOCAP/feature/denseface_openface_iemocap_mean_std_torch/img_db/fc
# img_dir=/data7/emobert/exp/evaluation/MSP/feature/denseface_openface_msp_mean_std_torch/ft_npzs/fc
# output_dir=/data7/emobert/exp/evaluation/MSP/feature/denseface_openface_msp_mean_std_torch/img_db/fc
# img_dir=/data7/emobert/exp/evaluation/MELD/feature/denseface_openface_meld_mean_std_torch/ft_npzs/fc
# output_dir=/data7/emobert/exp/evaluation/MELD/feature/denseface_openface_meld_mean_std_torch/img_db/fc
# img_dir=/data7/emobert/exp/evaluation/MELD/feature/denseface_openface_meld_mean_std_torch/raw_img_npzs
# output_dir=/data7/emobert/exp/evaluation/MELD/feature/denseface_openface_meld_mean_std_torch/raw_img_db
# img_dir=/data7/emobert/exp/evaluation/MSP/feature/openface_iemocap_raw_img/raw_img_npzs
# output_dir=/data7/emobert/exp/evaluation/MSP/feature/openface_iemocap_raw_img/raw_img_db
echo "convrting image features ..."
python convert_imgdir.py --img_dir $img_dir --output $output_dir --conf_th 1.0 --max_bb 360 --nproc 16
echo "done"