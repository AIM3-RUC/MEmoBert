import json
import os
import numpy as np

'''
将中文的数据整理为 ref_caption.json 文件
 "No0026.Mrs.Doubtfire_367": [
    "a small boy on some grass and a frisbee"
  ]
'''

transcripts_dir = '/data7/emobert/data_nomask/transcripts/json'
movie_names_path = '/data7/emobert/data_nomask/movies_v2/movie_names.npy'
save_path = '/data7/emobert/data_nomask/movies_v2/ref_captions.json'

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