import sys
import argparse
from ast import dump
import glob
import io
import json
import multiprocessing as mp
import os
from os.path import basename, exists, join
from token import NOTEQUAL

import cv2
from code.uniter.scripts.convert_imgdir import main
from preprocess.tasks.vision import FaceSelector
from preprocess.FileOps import read_file

import faulthandler
faulthandler.enable()

from cytoolz import curry
import numpy as np
from tqdm import tqdm
import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

'''
将raw-img，即densenet的输入，做为特征进行编码，这时候不需要soft-label了。
Step1: 将所有图片经过预处理保存为: (64,64) .npz
Step2: 构建rawimg_db库文件
'''

def prepocess_img(img_path):
    # movies_v1's mean and std
    mean = 63.987095
    std = 43.00519
    img_size = 64
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if not isinstance(img, np.ndarray):
            print('Warning: Error in {}'.format(img_path))
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_size, img_size))
        img = (img - mean) / std
        return img
    else:
        return None

def trans_img_npzs(face_dir, rawimg_npzs_dir, meta_data_dir, movie_names_path, start=0, end=0):
    '''
    fileter_dict: 未来加很多数据的时候可能
    segment_id = movie_name + '_' + segment_index
    No0123.Brokeback.Mountain_1.npz
    '''
    faceselector = FaceSelector()
    valid_movie_names = np.load(movie_names_path)
    print('total valid movies {} and start {} end {}'.format(len(valid_movie_names), start, end))
    for movie_name in valid_movie_names[start:end]:
        active_spk_filepath = os.path.join(meta_data_dir, movie_name, 'has_active_spk.txt')
        active_spk_lines = read_file(active_spk_filepath)
        count = 0
        for as_line in active_spk_lines:
            # such as: No0002.The.Godfather/6
            mn, segment_ind = as_line.strip().split('/')
            assert mn == movie_name
            outputfilename = movie_name + '_' + segment_ind + ".npz"
            outputfile = join(rawimg_npzs_dir, outputfilename)
            if os.path.exists(outputfile):
                continue
            active_spk = open(join(face_dir, movie_name, segment_ind, 'activate_spk.txt')).read().strip()
            assert active_spk != "None"
            active_spk = int(active_spk)
            face_dir_path = join(face_dir, movie_name, segment_ind)
            infos = faceselector(face_dir_path, active_spk)
            imgs = [x['img'] for x in infos]
            frame_nums = [x['frame_num'] for x in infos]
            confidence = [x['confidence'] for x in infos]
            rawimg_fts = []
            for img_path in imgs:
                # print(img_path)
                rawimg_ft = prepocess_img(img_path)
                if rawimg_ft is not None:
                    rawimg_fts.append(rawimg_ft)
                else:
                    print('[Error] {} is not exist'.format(img_path))
            # print(len(imgs), len(rawimg_fts), len(confidence), len(frame_nums))
            assert len(imgs) == len(rawimg_fts) == len(confidence) == len(frame_nums)
            rawimg_fts = np.array(rawimg_fts)
            confidence = np.array(confidence)
            frame_indexs = np.array(frame_nums)
            # print('rawimg_fts {} {} {}'.format(rawimg_fts.shape, confidence.shape, frame_indexs.shape))
            # print(frame_indexs)
            # print(confidence)
            # print(rawimg_fts[0].shape)
            np.savez(outputfile,
                            frame_idxs=frame_indexs.astype(np.float16),
                            confidence=confidence.astype(np.float16),
                            features=rawimg_fts.astype(np.float16))
            # print('Save is OK')
            count += 1
        print('Cur {} and total videos {}'.format(movie_name, count))


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    if dataset_name == 'iemocap':
        mean, std = 131.0754, 47.858177
    elif dataset_name == 'msp':
        mean, std = 96.3801, 53.615868
    elif dataset_name == 'meld':
        mean, std = 67.61417, 37.89171
    else:
        print('the dataset name is error {}'.format(dataset_name))
    raw_npzs_dir = '/data8/emobert/rawimg_npzs_nomask/' 
    meta_data_dir = '/data7/emobert/data_nomask/meta'
    movie_names_path = '/data7/emobert/data_nomask/movies_v3/'
    if not os.path.exists(raw_npzs_dir):
        os.makedirs(raw_npzs_dir)
    trans_img_npzs(face_dir, raw_npzs_dir, meta_data_dir, movie_names_path)