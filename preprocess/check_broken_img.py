import cv2
import os, glob
import numpy as np
from tqdm import tqdm
from numpy.lib.arraysetops import isin
from tasks.vision import FaceSelector

def get_all_imgs(dir_path, face_selector):
    active_spk = open(os.path.join(dir_path, 'activate_spk.txt')).read().strip()
    assert active_spk != "None", dir_path
    active_spk = int(active_spk)
    imgs = face_selector(dir_path, active_spk)
    return imgs

if __name__ == '__main__':
    meta_dir = 'data/meta'
    face_dir = 'data/faces'
    txt_name = 'has_active_spk.txt'
    face_selector = FaceSelector()

    all_txts = glob.glob(os.path.join(meta_dir, '*', txt_name))
    prob_set = set()
    for txt in tqdm(all_txts):
        utt_ids = open(txt).readlines()
        utt_ids = list(map(lambda x: x.strip(), utt_ids))
        for utt_id in utt_ids:
            all_imgs = get_all_imgs(os.path.join(face_dir, utt_id), face_selector)
            for img in all_imgs:
                data = cv2.imread(img)
                if not isinstance(data, np.ndarray):
                    movie_name = txt.split('/')[-2]
                    prob_set.add(movie_name)
    
    for x in prob_set:
        print(x)
