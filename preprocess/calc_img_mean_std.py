import os, glob
import cv2
import numpy as np
from tqdm import tqdm
from preprocess.FileOps import read_file, write_pkl

def select_pics(num_per_video=4000):
    root = 'data/faces'
    total_videos = os.listdir(root)
    all_pics = []
    for video in tqdm(total_videos):
        video_pics = sorted(glob.glob(os.path.join(root, video, '*/*/*.bmp')))
        total_length = len(video_pics)
        if not total_length:
            continue
        index = [int(x*total_length/num_per_video) for x in range(num_per_video)]
        all_pics += [video_pics[x] for x in index] 

    return all_pics

def select_pics_iemocap():
    all_pic = glob.glob('/data7/MEmoBert/evaluation/IEMOCAP/Session*/face/*/*.jpg')
    print(all_pic[:100])
    return all_pic


def calc_mean_std():
    # all_pics = select_pics()
    all_pics = select_pics_iemocap()
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
    # np.save('4k_per_video_mean.npy', mean)
    # np.save('4k_per_video_std.npy', std)

calc_mean_std()
# select_pics_iemocap()