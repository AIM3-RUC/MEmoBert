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

def calc_mean_std():
    all_pics = select_pics()
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
    np.save('4k_per_video_mean.npy', mean)
    np.save('4k_per_video_std.npy', std)


def gen_imgs_means_std(face_dir_list_path, filter='DE', normbyspeaker_path=None):
    '''
    normbyspeaker_path by all images
    :param face_dir_list_path:
    :param dsfd_face_dir_list is another face extractor
    :param filter: DE / HU / CN
    :return:
    '''
    import random
    vid2mean_std = {}
    images = []
    face_dir_lines = read_file(face_dir_list_path)
    for line in face_dir_lines:
        video_images = []
        video_face_dir = line.strip()
        videoId = os.path.split(video_face_dir)[1]
        if filter is None or filter in videoId:
            print(video_face_dir)
            imgs = os.listdir(video_face_dir)
            try:
                imgs = random.sample(imgs, 1000)
            except:
                imgs = imgs
            for img in imgs:
                if 'jpg' in img or 'png' in img:
                    img_path = os.path.join(video_face_dir, img)
                    # 如果图片太小直接过滤掉.
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (64, 64))
                    if np.sum(img) > 0:
                        images.append(img)
                        video_images.append(img)
            if normbyspeaker_path is not None:
                mean = np.mean(video_images)
                std = np.std(video_images)
                vid2mean_std[videoId] = [mean, std]
                print(videoId, mean, std)
    if normbyspeaker_path is not None:
        write_pkl(normbyspeaker_path, vid2mean_std)
    mean = np.mean(images)
    std = np.std(images)
    print("CES Face filter {} images {} mean {} std {}".format(filter, len(images), mean, std))

calc_mean_std()