# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
set -e
img_dir=/data7/emobert/ft_npzs/movie110_v1
output_dir=/data7/emobert/img_db
echo "converting image features ..."
python convert_imgdir.py --img_dir $img_dir --output $output_dir --nproc 32
echo "done"