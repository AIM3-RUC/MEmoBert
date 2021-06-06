import json
import os
import numpy as np

'''
将中文的数据整理为 ref_caption.json 文件
 "No0026.Mrs.Doubtfire_367": [
    "a small boy on some grass and a frisbee"
  ]
'''

if False:
  ## for moivies data
  transcripts_dir = '/data7/emobert/data_nomask/transcripts/json'
  movie_names_path = '/data7/emobert/data_nomask/movies_v3/movie_names.npy'
  save_path = '/data7/emobert/data_nomask/movies_v3/ref_captions.json'
  ref_captions = {}
  movie_names = np.load(movie_names_path)
  for movie_name in movie_names:
      json_path = os.path.join(transcripts_dir, movie_name + '.json')
      data = json.load(open(json_path))
      for segment_index in data.keys():
          segmentId = movie_name + '_' + segment_index
          ref_captions[segmentId] = [data[segment_index]['content']]
  print('total captions {}'.format(len(ref_captions)))
  with open(save_path, 'w') as f:
      json.dump(ref_captions, f)

if True:
  ## for voxceleb2-v1 data
  # 构建有效的ID文件, id07367#2m55mUzafEs_00004, 用#号分割作为链接符号，/很容易出现问题
  # feat_dir = '/data13/voxceleb2/denseface_feature'
  # movie_names_path = '/data7/emobert/data_nomask_new/voxceleb2_v1/movie_names.npy'
  # movie_names = []
  # for spkId in os.listdir(feat_dir):
  #   cur_dir = os.path.join(feat_dir, spkId)
  #   for videoId in os.listdir(cur_dir):
  #     movie_names.append(spkId + '#' + videoId)
  # print(f'movie_names {len(movie_names)}')
  # np.save(movie_names_path, movie_names)

  ## for voxceleb2-v2 data
  # feat_dir = '/data13/voxceleb2/denseface_feature'
  # v1_movie_names_path = '/data7/emobert/data_nomask_new/voxceleb2_v1/movie_names.npy'
  # v1_movie_names = np.load(v1_movie_names_path)
  # v1_spkIds = list(set(m.split('#')[0] for m in v1_movie_names))
  # print('v1 spkIds {}'.format(len(v1_spkIds)))
  # movie_names_path = '/data7/emobert/data_nomask_new/voxceleb2_v2/movie_names.npy'
  # movie_names = []
  # for spkId in os.listdir(feat_dir):
  #   if spkId in v1_spkIds:
  #       continue
  #   cur_dir = os.path.join(feat_dir, spkId)
  #   for videoId in os.listdir(cur_dir):
  #     movie_names.append(spkId + '#' + videoId)
  # print(f'movie_names {len(movie_names)}')
  # np.save(movie_names_path, movie_names)

  json_path = '/data10/voxceleb2/store/sentence_map.json'
  movie_names_path = '/data7/emobert/data_nomask_new/voxceleb2_v2/movie_names.npy'
  save_path = '/data7/emobert/data_nomask_new/voxceleb2_v2/ref_captions.json'
  ref_captions = {}
  movie_names = np.load(movie_names_path)
  movie_names_dict = {m:1 for m in movie_names}
  data = json.load(open(json_path))
  for key in data.keys():
    splits = key.split('/')
    movie_name = splits[0] + '#' + splits[1]
    if movie_names_dict.get(movie_name) is not None:
      segmentId = movie_name + '#' + splits[2]
      ref_captions[segmentId] = [' '.join(data[key])]
  print('total captions {}'.format(len(ref_captions)))
  with open(save_path, 'w') as f:
      json.dump(ref_captions, f)