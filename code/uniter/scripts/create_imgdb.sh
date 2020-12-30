# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
set -e
# img_dir=/data7/emobert/ft_npzs_nomask/movies_v1
# output_dir=/data7/emobert/img_db_nomask
# img_dir=/data7/emobert/exp/evaluation/IEMOCAP/feature/denseface_openface_mean_std_movie_no_mask/ft_npzs/iemocap
img_dir=/data7/emobert/exp/evaluation/MSP-IMPROV/feature/denseface_seetaface_mean_std_movie_no_mask/ft_npzs/msp
output_dir=/data7/emobert/exp/evaluation/MSP-IMPROV/feature/denseface_seetaface_mean_std_movie_no_mask/img_db
echo "converting image features ..."
python convert_imgdir.py --img_dir $img_dir --output $output_dir --conf_th 0.0 --nproc 32
echo "done"