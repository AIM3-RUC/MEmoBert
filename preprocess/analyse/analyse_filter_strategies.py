import h5py
import os

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

ori_info_dir = '/data7/emobert/data/transcripts/json'
meta_data_dir = '/data7/emobert/data/meta'
feature_dir = '/data7/emobert/feature'
fileter_details_path = '/data7/emobert/data/analyse/fileter_details.txt'

total_lines = 0 
final_lines = 0
total_words_len = 0
total_faces_len = 0
details_lines = []

movie_list = os.listdir(ori_info_dir)
print('there are total {} movies'.format(len(movie_list)))
for filename in movie_list:
    ori_segments2info = read_json(os.path.join(ori_info_dir, filename))
    movie_name = filename[:-5]  # remove '.json'
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
        print('{} movie is empty, Please Check this'.format(movie_name))
        continue
    total_lines += len(ori_segments2info)
    final_lines += len(active_spk_lines)
    # get the content words info
    movie_words_lens = []
    for line in active_spk_lines:
        # No0040.Persuasion/2
        segment_index = line.strip('\n').split('/')[-1]
        text = ori_segments2info[segment_index]['content']
        _len = len(text.split(' '))
        movie_words_lens.append(_len)
    assert len(active_spk_lines) == len(movie_words_lens)
    total_words_len += sum(movie_words_lens)
    details_lines.append('\t Average words {} \t {:.2f}'.format(len(movie_words_lens), sum(movie_words_lens)/len(movie_words_lens))+ '\n')
    # get the face frames info
    movie_faces_lens = []
    denseface_ft_path = os.path.join(feature_dir, movie_name, 'has_active_spk_denseface.h5')
    denseface_ft = h5py.File(denseface_ft_path)
    for segment_index in denseface_ft[movie_name].keys():
        _len = denseface_ft[movie_name][segment_index]['pred'].shape[0]
        movie_faces_lens.append(_len)
    assert len(active_spk_lines) == len(movie_faces_lens)
    total_faces_len += sum(movie_faces_lens)
    details_lines.append('\t Average faces {} \t {:.2f}'.format(len(movie_faces_lens), sum(movie_faces_lens)/len(movie_faces_lens))+ '\n')
details_lines.append('Toal {} {} and {:.2f} and avg words {:.2f} avg faces {:.2f}'.format(total_lines, final_lines, final_lines / total_lines, \
                 total_words_len/final_lines, total_faces_len/final_lines))
write_file(fileter_details_path, details_lines)