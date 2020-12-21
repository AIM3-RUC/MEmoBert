'''
获取需要抽取的uttid
'''

import os
import numpy as np

if __name__ == '__main__':
    utt_id_root = '/data3/lrc/Iemocap_feature/session_level/target'
    utt_ids = []
    for i in range(1, 6):
        sess_int2name = np.load(os.path.join(utt_id_root, \
                        'iemocap_session{}_int2name.npy'.format(i)))
        utt_ids.append(sess_int2name)
    utt_ids = np.concatenate(utt_ids)
    print(utt_ids.shape)
    utt_ids = utt_ids.tolist()
    utt_ids = list(map(lambda x: x.decode()+'\n', utt_ids))
    f = open('utt_ids.txt', 'w')
    f.writelines(utt_ids)