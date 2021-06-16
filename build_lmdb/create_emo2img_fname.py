from os import posix_fadvise
import pickle as pkl
import numpy as np
import h5py
import json
'''
根据文本得到的情感类别进行构建负例，在txtdb的目录下构建一个 emo2img_fname.pkl的文件，顺便统计一个目前的情感分布。
/data7/emobert/txt_pseudo_label/movie_txt_pseudo_label_v1.h5
/data7/emobert/txt_pseudo_label/movie_v1_emo2img_fname.pkl
movies_v1: emo-0 : 29406 emo-2 : 7657 emo-1 : 8382 emo-4 : 8439 emo-3 : 7000
movies_v2: emo-1 : 3408  emo-0 : 17375 emo-4 : 3839 emo-3 : 3706 emo-2 : 3820
movies_v3: emo-0 : 46707 emo-1 : 11848 emo-3 : 9700 emo-4 : 12349 emo-2 : 11558
voxceleb2_v1 emo-0 : 369921 emo-1 : 9000 emo-4 : 852 emo-2 : 100 emo-3 : 3341
voxceleb2_v2  emo-0 : 218885 emo-1 : 5488 emo-3 : 1981 emo-4 : 528 emo-2 : 56

all_5corpus_emo5:
movies_v1： emo-0: 33608 emo-1: 10627 emo-2: 4017 emo-3: 5108 emo-4: 7524
movies_v2： emo-0: 19340 emo-1: 4546  emo-2: 1957 emo-3: 2874 emo-4: 3431
movies_v3： emo-0: 50910 emo-1: 15617 emo-2: 6297 emo-3: 7718 emo-4: 11620
'''

# version, v1, v2, or v3
version = 'v3'
postfix = 'all_5corpus_emo5'
# all_emo2img_fname = f'/data7/emobert/txt_pseudo_label/movie_{version}_emo2img_fname_{postfix}.pkl'
# all_emo2img_fname_dict = {}
# all_text2img_path = f'/data7/emobert/txt_db/movies_{version}_th0.5_emowords_sentiword_all.db/txt2img.json'
# all_targe_path = f'/data7/emobert/txt_pseudo_label/movie_txt_pseudo_label_{version}_{postfix}.h5'

all_emo2img_fname = '/data7/emobert/txt_pseudo_label/voxceleb2_{}_emo2img_fname_{postfix}.pkl'
all_emo2img_fname_dict = {}
all_text2img_path = '/data7/emobert/txt_db/voxceleb2_v2_th1.0_emowords_sentiword_all.db/txt2img.json'
all_targe_path = '/data7/emobert/txt_pseudo_label/voxceleb2_txt_pseudo_label_v2.h5'

all_textId2target = h5py.File(all_targe_path, 'r')
all_text2img = json.load(open(all_text2img_path))
assert len(all_textId2target.keys()) == len(all_text2img)
# transfer to all imgId2target
imgId2target = {}
for textId in all_text2img.keys():
    img_fname = all_text2img[textId]
    target = all_textId2target[textId]
    imgId2target[img_fname] = target
    emo_cate = np.argmax(target['pred'])
    if all_emo2img_fname_dict.get(emo_cate) is None:
        all_emo2img_fname_dict[emo_cate] = [img_fname]
    else:
        all_emo2img_fname_dict[emo_cate] += [img_fname]
print('all_text2img {}'.format(len(all_text2img)))
for emo_cate in all_emo2img_fname_dict.keys():
    img_fnames = all_emo2img_fname_dict[emo_cate]
    print('emo-{}: {}'.format(emo_cate, len(img_fnames)))
pkl.dump(all_emo2img_fname_dict, open(all_emo2img_fname, 'wb'))