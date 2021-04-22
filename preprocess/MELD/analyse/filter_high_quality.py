import numpy
from preprocess.FileOps import read_json, write_json

'''
case1: 只保留大约一张图片的sampels, 包括所有的训练集合和测试集合
重新构建文本集合, 根据如下frames的数目进行筛选:
/data7/MEmoBert/emobert/exp/evaluation/MELD/feature/denseface_openface_meld_mean_std_torch/raw_img_db/nbb_th0.5_max36_min10.json
train total keys 9989 there are 492 less than 2 frames
val total keys 1109 there are 59 less than 2 frames
test total keys 2610 there are 138 less than 2 frames

case2: 只保留具有 activate-spk 的 samples
'''

def get_none_frame_videos(setname):
    text_info = f'/data7/MEmoBert/emobert/exp/evaluation/MELD/txt_db/1/{setname}_emowords_emotype.db/img2txts.json'
    text_data = read_json(text_info)
    json_path = '/data7/MEmoBert/emobert/exp/evaluation/MELD/feature/denseface_openface_meld_mean_std_torch/raw_img_db/nbb_th0.5_max36_min10.json'
    data = read_json(json_path)
    set_keys = list(text_data.keys())
    print('{} total keys {}'.format(setname, len(set_keys)))
    count = 0
    save_keys = {}
    save_keys_path = '/data7/MEmoBert/emobert/exp/evaluation/MELD/txt_db/{}_low_quality_less2.json'.format(setname)
    for key in data.keys():
        if data[key] < 2:
            if key in set_keys:
                count += 1
                save_keys[key] = 1
    write_json(save_keys_path, save_keys)
    print('there are {} less than 2 frames'.format(len(save_keys)))


if __name__ == '__main__':
    get_none_frame_videos(setname='train')
    get_none_frame_videos(setname='val')
    get_none_frame_videos(setname='test')