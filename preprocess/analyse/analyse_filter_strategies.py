import h5py
import os
import numpy as np
from numpy.lib.function_base import append

from numpy.testing._private.utils import assert_raises
from preprocess.FileOps import read_json, read_file, write_file

'''
export PYTHONPATH=/data7/MEmoBert
分析过滤机制中数据的
/data7/emobert/data/meta/No0092.Wonder/:
base.txt: 是否句子长度大于等于1并且时间长度大于等于1秒
has_face.txt: 先去除完全没有人脸的片段
longest_spk_0.2.txt: 出现时间最长的人，占视频长度的20，保证有人出现。
has_active_spk.txt: 是否包含说话人，说话人的条件比较严格。
统计平均的句子长度 以及 平均的脸的帧数
'''

ori_info_dir = '/data7/emobert/data_nomask/transcripts/json'
meta_data_dir = '/data7/emobert/data_nomask/meta'
feature_dir = '/data7/emobert/feature_nomask_torch'
fileter_details_path = '/data7/emobert/data_nomask/analyse/fileter_details_moviesv3.txt'
filter_movie_names_path = ['/data7/emobert/data_nomask/movies_v1/movie_names.npy', 
    '/data7/emobert/data_nomask/movies_v2/movie_names.npy']
movie_names_path = '/data7/emobert/data_nomask/movies_v3/movie_names.npy'

total_lines = 0 
final_lines = 0
total_words_lens = []
total_faces_lens = []
details_lines = []
valid_movies = []

def compute_stastic_info(lens):
    # 返回长度的中位数和80%分位点
    lens.sort()
    avg_len = sum(lens) / len(lens)
    mid_len = lens[int(len(lens)/2)]
    m80_len = lens[int(len(lens)*0.8)]
    return avg_len, mid_len, m80_len

if filter_movie_names_path is not None:
    filter_movie_names = {}
    for filter_path in filter_movie_names_path:
        names = np.load(filter_path)
        filter_names = {n:1 for n in names}
        filter_movie_names.update(filter_names)
    print('There are {} movies have processed'.format(len(filter_movie_names)))
else:
    filter_movie_names = {}

movie_list = os.listdir(ori_info_dir)
print('there are total {} movies'.format(len(movie_list)))
for filename in movie_list:
    ori_segments2info = read_json(os.path.join(ori_info_dir, filename))
    movie_name = filename[:-5]  # remove '.json'
    if filter_movie_names.get(movie_name) is not None:
        continue
    movie_meta_dir = os.path.join(meta_data_dir, movie_name)
    # base filter 
    base_filepath = os.path.join(movie_meta_dir, 'base.txt')
    if not os.path.exists(base_filepath): # 有的电影字幕文件不对或者没有字幕
        continue
    base_lines = read_file(base_filepath)
    details_lines.append('Cur {} {}'.format(movie_name, len(ori_segments2info)) + '\n')
    details_lines.append('\t Base {} \t {:.2f}'.format(len(base_lines), len(base_lines)/len(ori_segments2info))+ '\n')
    has_face_filepath = os.path.join(movie_meta_dir, 'has_face.txt')
    has_face_lines = read_file(has_face_filepath)
    details_lines.append('\t HasFace {} \t {:.2f}'.format(len(has_face_lines), len(has_face_lines)/len(ori_segments2info))+ '\n')
    long_face_filepath = os.path.join(movie_meta_dir, 'longest_spk_0.2.txt')
    long_face_lines = read_file(long_face_filepath)
    details_lines.append('\t LongFace {} \t {:.2f}'.format(len(long_face_lines), len(long_face_lines)/len(ori_segments2info))+ '\n')
    active_spk_filepath = os.path.join(movie_meta_dir, 'has_active_spk.txt')
    active_spk_lines = read_file(active_spk_filepath)
    details_lines.append('\t ActiveSpk {} \t {:.2f}'.format(len(active_spk_lines), len(active_spk_lines)/len(ori_segments2info))+ '\n')
    if len(has_face_lines) == 0:
        print('\tMovie is empty{} , Please Check this'.format(movie_name))
        continue
    #### get the face frames info
    movie_faces_lens = []
    denseface_ft_path = os.path.join(feature_dir, movie_name, 'has_active_spk_denseface_with_trans.h5')
    if not os.path.exists(denseface_ft_path):
        print('\t We ignore this moive {}'.format(movie_name))
        continue
    total_lines += len(ori_segments2info)
    final_lines += len(active_spk_lines)
    print("---- Valid {}".format(movie_name))
    denseface_ft = h5py.File(denseface_ft_path)
    for segment_index in denseface_ft[movie_name].keys():
        _len = denseface_ft[movie_name][segment_index]['pred'].shape[0]
        movie_faces_lens.append(_len)
    total_faces_lens.extend(movie_faces_lens)
    assert len(active_spk_lines) == len(movie_faces_lens)
    avg_len, min_len, m80_len = compute_stastic_info(movie_faces_lens)
    details_lines.append('\t Face {} Avg {:.2f} Mid {:.2f} Mid80 {:.2f}'.format(len(movie_faces_lens), avg_len, min_len, m80_len) + '\n')
    # get the content words info
    movie_words_lens = []
    for line in active_spk_lines:
        # No0040.Persuasion/2
        segment_index = line.strip('\n').split('/')[-1]
        text = ori_segments2info[segment_index]['content']
        _len = len(text.split(' '))
        movie_words_lens.append(_len)
    total_words_lens.extend(movie_words_lens)
    assert len(active_spk_lines) == len(movie_words_lens)
    avg_len, min_len, m80_len = compute_stastic_info(movie_faces_lens)
    details_lines.append('\t Sents {} Avg {:.2f} Mid {:.2f} Mid80 {:.2f}'.format(len(movie_words_lens), avg_len, min_len, m80_len) + '\n')
    valid_movies.append(movie_name)
details_lines.append('total vilid {} movies'.format(len(valid_movies)))
details_lines.append('Toal {} and valid {}'.format(total_lines, final_lines))
avg_len, min_len, m80_len = compute_stastic_info(total_faces_lens)
details_lines.append('Face {} avg {:.2f} mid {:.2f} mid80 {:.2f}'.format(len(total_faces_lens), avg_len, min_len, m80_len))
avg_len, min_len, m80_len = compute_stastic_info(total_words_lens)
details_lines.append('Sents {} avg {:.2f} mid {:.2f} mid80 {:.2f}'.format(len(total_words_lens), avg_len, min_len, m80_len))
write_file(fileter_details_path, details_lines)
np.save(movie_names_path, valid_movies)