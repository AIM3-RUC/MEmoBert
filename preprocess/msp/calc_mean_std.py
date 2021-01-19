import os, glob
import cv2
import numpy as np
from tqdm import tqdm

def select_pics_msp():
    if name == 'seetaface':
        all_pic = glob.glob('/data7/emobert/exp/evaluation/MSP-IMPROV/Face/*/*.jpg')
    elif name == 'openface':
        all_pic = glob.glob('/data7/emobert/exp/evaluation/MSP-IMPROV/OpenFace/*/*_aligned/*.bmp')
    else:
        raise ValueError('Must be seetaface or openface')
    print(all_pic[:10])
    input()
    return all_pic


def calc_mean_std():
    # all_pics = select_pics()
    all_pics = select_pics_msp()
    data = []
    for pic in tqdm(all_pics):
        _d = cv2.imread(pic)
        if _d is None:
            continue
        _d = cv2.cvtColor(_d, cv2.COLOR_BGR2GRAY)
        _d = cv2.resize(_d, (64, 64))
        data.append(_d)
    
    data = np.array(data).astype(np.float32)
    print('Total Data Shape:', data.shape)
    mean = np.mean(data)
    std = np.std(data)
    print('Mean:', mean)
    print('Std:', std)

import sys
name = sys.argv[1]
calc_mean_std()

# seetaface
# Mean: 113.59194
# Std: 45.201733
# openface
# Mean: 113.30168
# Std: 42.49455