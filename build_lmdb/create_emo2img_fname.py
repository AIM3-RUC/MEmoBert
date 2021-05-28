import pickle as pkl
import numpy as np
import h5py
import json
'''
根据文本得到的情感类别进行构建负例，在txtdb的目录下构建一个 emo2img_fname.pkl的文件，顺便统计一个目前的情感分布。
/data7/emobert/txt_pseudo_label/movie_txt_pseudo_label_v1.h5
/data7/emobert/txt_pseudo_label/movie_v1_emo2img_fname.pkl
movies_v1: emo 0 and fnames 29406 emo 2 and fnames 7657 emo 1 and fnames 8382 emo 4 and fnames 8439 emo 3 and fnames 7000
movies_v2: emo 1 and fnames 3408  emo 0 and fnames 17375 emo 4 and fnames 3839 emo 3 and fnames 3706 emo 2 and fnames 3820
movies_v3: emo 0 and fnames 46707 emo 1 and fnames 11848 emo 3 and fnames 9700 emo 4 and fnames 12349 emo 2 and fnames 11558
voxceleb2_v1 emo 0 and fnames 369921 emo 1 and fnames 9000 emo 4 and fnames 852 emo 2 and fnames 100 emo 3 and fnames 3341
'''

# version, v1, v2, or v3
# version = 'v3'
# all_emo2img_fname = f'/data7/emobert/txt_pseudo_label/movie_{version}_emo2img_fname.pkl'
# all_emo2img_fname_dict = {}
# all_text2img_path = f'/data7/emobert/txt_db/movies_{version}_th0.5_emowords_sentiword_all.db/txt2img.json'
# all_targe_path = f'/data7/emobert/txt_pseudo_label/movie_txt_pseudo_label_{version}.h5'

all_emo2img_fname = '/data7/emobert/txt_pseudo_label/voxceleb2_v1_emo2img_fname.pkl'
all_emo2img_fname_dict = {}
all_text2img_path = '/data7/emobert/txt_db/voxceleb2_v1_th1.0_emowords_sentiword_all.db//txt2img.json'
all_targe_path = '/data7/emobert/txt_pseudo_label/voxceleb2_txt_pseudo_label_v1.h5'
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
    print('emo {} and fnames {}'.format(emo_cate, len(img_fnames)))
pkl.dump(all_emo2img_fname_dict, open(all_emo2img_fname, 'wb'))