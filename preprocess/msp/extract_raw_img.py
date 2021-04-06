import os, sys, glob
import cv2
import tensorflow as tf
import numpy as np
import collections
from tqdm import tqdm
import traceback

class FeatureExtractor(object):
    def __init__(self, mean, std, smooth=False):
        """ extract densenet feature
            Parameters:
            ------------------------
            model: model class returned by function 'load_model'
        """
        self.mean = mean
        self.std = std
        self.previous_img = None        # smooth 的情况下, 如果没有人脸则用上一张人脸填充
        self.previous_img_path = None
        self.smooth = smooth
    
    def extract_feature(self, img_path):
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (Img_dim, Img_dim))
            if self.smooth:
                self.previous_img = img
                self.previous_img_path = img_path
        elif self.smooth and self.previous_img is not None:
            # print('Path {} does not exists. Use previous img: {}'.format(img_path, self.previous_img_path))
            img = self.previous_img
        else:
            img = np.zeros([Img_dim, Img_dim]) # smooth的话第一张就是黑图的话就直接返回0特征, 不smooth缺图就返回0
            return img
        img = (img - self.mean) / self.std
        return img

def make_denseface(): 
    mean = 96.3801
    std = 53.615868
    smooth = False
    total_video = 0
    extractor = FeatureExtractor(mean=mean, std=std, smooth=smooth)
    for videoid in os.listdir(face_root):
        outputfilename = videoid + ".npz"
        outputfile = os.path.join(output_dir, outputfilename)
        total_video += 1
        if os.path.exists(outputfile):
            continue
        print(outputfile)
        frame_dir = os.path.join(frame_root, videoid)
        face_dir = os.path.join(face_root, videoid)
        if not os.path.exists(face_dir):
            raise RuntimeError('Face should be extract first for video:{}'.format(videoid))

        ans = []
        frame_pics = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')), key=lambda x: int(os.path.basename(x).split('.')[0]))
        if len(frame_pics) == 0:
            print('[Error] there are no frames')
        for frame_pic in frame_pics:
            basename = os.path.basename(frame_pic)
            face_path = os.path.join(face_dir, basename)
            # print(face_path)
            feat = extractor.extract_feature(face_path)
            ans.append(feat)
        ans = np.array(ans, dtype=np.float32)
        print(videoid, ans.shape)
        np.savez(outputfile,
                            features=ans.astype(np.float16))    
    print('total videos {}'.format(total_video))

if __name__ == '__main__':
    import sys
    Img_dim = 112
    output_dir = '/data7/emobert/exp/evaluation/MSP/feature/openface_iemocap_raw_img/raw_img_npzs'
    face_root = '/data7/emobert/exp/evaluation/MSP/Face'
    frame_root = '/data7/emobert/exp/evaluation/MSP/Frame'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    make_denseface()