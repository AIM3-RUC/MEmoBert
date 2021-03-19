import json

def compute_stastic_info(lens):
    # 返回长度的中位数和80%分位点
    lens.sort()
    avg_len = sum(lens) / len(lens)
    mid_len = lens[int(len(lens)/2)]
    m80_len = lens[int(len(lens)*0.8)]
    return avg_len, mid_len, m80_len

# lmdb_name = 'movies_v2/fc'
# filepath = '/data7/emobert/img_db_nomask/{}/nbb_th0.5_max36_min10.json'.format(lmdb_name)
# filepath = '/data7/emobert/exp/evaluation/IEMOCAP/feature/denseface_openface_iemocap_mean_std_torch/img_db/fc/nbb_th0.0_max36_min10.json'
filepath = '/data7/emobert/exp/evaluation/MSP/feature/denseface_openface_msp_mean_std_torch/img_db/fc/nbb_th0.0_max64_min10.json'

video2lens = json.load(open(filepath))
movie_faces_lens = []
for key in video2lens.keys():
    _len = video2lens[key]
    movie_faces_lens.append(_len)
avg_len, min_len, m80_len = compute_stastic_info(movie_faces_lens)
# print(movie_faces_lens[0:30])
# print(movie_faces_lens[540:600])
print('\t Face {} Avg {:.2f} Mid {:.2f} Mid80 {:.2f}'.format(len(movie_faces_lens), avg_len, min_len, m80_len) + '\n')