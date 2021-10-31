import collections
import os
import random
import numpy as np
from preprocess.FileOps import read_file, read_pkl, write_pkl

'''
export PYTHONPATH=/data9/MEmoConv
整理为numpy的方式 可以用于当前的模型:
/data9/memoconv/modality_fts/utt_baseline
    --{setname}/int2name.npy
    --{setname}/label.npy
    --{setname}/audio_{}_ft.npy
    --{setname}/text_{}_ft.npy
    --{setname}/face_{}_ft.npy
'''

def get_set_label_info(set_movie_names_filepath, target_dir):
    int2label, int2name = [], []
    set_movie_names = read_file(set_movie_names_filepath)
    set_movie_names = [name.strip() for name in set_movie_names]
    for movie_name in set_movie_names:
        movie_label_filepath = os.path.join(target_dir, '{}_label.pkl'.format(movie_name))
        uttId2label = read_pkl(movie_label_filepath)
        for uttId in uttId2label.keys():
            int2name.append(uttId)
            int2label.append(uttId2label[uttId])
    assert len(int2name) == len(int2label)
    return int2name, int2label

def compute_stastic_info(lens):
    # 返回长度的中位数和80%, 95%分位点
    lens.sort()
    avg_len = sum(lens) / len(lens)
    mid_len = lens[int(len(lens)/2)]
    m80_len = lens[int(len(lens)*0.8)]
    m95_len = lens[int(len(lens)*0.95)]
    return avg_len, mid_len, m80_len, m95_len

def get_set_ft_info(set_movie_names_filepath, output_int2name_filepath, modality_ft_dir, modality, feature_type, feature_dim):
    set_fts = []
    ft_lens = []
    set_movie_names = read_file(set_movie_names_filepath)
    set_movie_names = [name.strip() for name in set_movie_names]
    int2name = np.load(output_int2name_filepath)
    int2name_dict = {name:1 for name in int2name}
    for movie_name in set_movie_names:
        movie_modality_ft_filepath = os.path.join(modality_ft_dir, '{}_{}_ft_{}.pkl'.format(movie_name, modality, feature_type))
        assert os.path.exists(movie_modality_ft_filepath) == True
        uttId2ft = read_pkl(movie_modality_ft_filepath)
        # spk错误case的修正
        if uttId2ft.get('A_jimaofeishangtian_13_6') is not None:
            print('Modify the spk error case A_jimaofeishangtian_13_6')
            uttId2ft['B_jimaofeishangtian_13_6'] = uttId2ft['A_jimaofeishangtian_13_6']
        for uttId in uttId2ft:
            if int2name_dict.get(uttId) is not None:
                if len(uttId2ft[uttId].shape) == 3:
                    uttId2ft[uttId] = uttId2ft[uttId][0]
                # 异常情况处理
                if uttId2ft[uttId].shape[-1] == 0:
                    print('Utt {} is Feature Dim is 0'.format(uttId))
                    uttId2ft[uttId] = np.zeros([1, feature_dim])
                # 异常情况处理
                if uttId2ft[uttId].size == 1:
                    print('Utt {} is None Speech/Visual'.format(uttId))
                    uttId2ft[uttId] = np.zeros([1, feature_dim])
                set_fts.append(uttId2ft[uttId])
                ft_lens.append(len(uttId2ft[uttId]))
    avg_len, mid_len, m80_len, m95_len = compute_stastic_info(ft_lens)
    print('int2name {} {}'.format(len(int2name), len(set_fts)))
    assert len(int2name) == len(set_fts)
    print(f'{modality} {feature_type} avg {avg_len} mid {mid_len} p80 {m80_len} p95 {m95_len}')
    return set_fts

def comparE_norm(feat_dir):
    mean_std_filepath = os.path.join(feat_dir, 'train', 'speech_comparE_mean0_std1.npy')
    train_ft_filepath = os.path.join(feat_dir, 'train', 'speech_comparE_ft.npy')
    train_fts = np.load(train_ft_filepath, allow_pickle=True)
    if os.path.exists(mean_std_filepath):
        print('existing mean and std')
        mean, std = np.load(mean_std_filepath)
    else:
        total = np.concatenate(train_fts, axis=0)
        select_indexs = np.linspace(0, len(total)-1, int(len(total)/4), dtype=int)
        print(len(select_indexs), select_indexs[:10])
        total = [total[i] for i in select_indexs]
        total = np.array(total, dtype=np.float32)
        for v in total:
            if v.shape[0] != 130:
                print(v.shape)
        print(total.shape)
        mean = np.mean(total, axis=0)
        std = np.std(total, axis=0)
        std[std==0.0] = 1.0
        np.save(mean_std_filepath, [mean, std])
    print('process train')
    norm_train_fts = []
    for ft in train_fts:
        norm_ft = (ft - mean) / std
        norm_train_fts.append(norm_ft)
    np.save(os.path.join(feat_dir, 'train', 'speech_comparE_norm_ft.npy'), norm_train_fts)
    print('process validation')
    val_ft_filepath = os.path.join(feat_dir, 'val', 'speech_comparE_ft.npy')
    val_fts = np.load(val_ft_filepath, allow_pickle=True)
    norm_val_fts = []
    for ft in val_fts:
        norm_ft = (ft - mean) / std
        norm_val_fts.append(norm_ft)
    np.save(os.path.join(feat_dir, 'val', 'speech_comparE_norm_ft.npy'), norm_val_fts)
    print('process test')
    test_ft_filepath = os.path.join(feat_dir, 'test', 'speech_comparE_ft.npy')
    test_fts = np.load(test_ft_filepath, allow_pickle=True)
    norm_test_fts = []
    for ft in test_fts:
        norm_ft = (ft - mean) / std
        norm_test_fts.append(norm_ft)
    np.save(os.path.join(feat_dir, 'test', 'speech_comparE_norm_ft.npy'), norm_test_fts)

def IS10_norm(feat_dir):
    mean_std_filepath = os.path.join(feat_dir, 'train', 'speech_IS10_mean0_std1.npy')
    train_ft_filepath = os.path.join(feat_dir, 'train', 'speech_IS10_ft.npy')
    train_fts = np.load(train_ft_filepath, allow_pickle=True)
    if os.path.exists(mean_std_filepath):
        print('existing mean and std')
        mean, std = np.load(mean_std_filepath)
    else:
        total = np.array(train_fts, dtype=np.float32)
        print(total.shape)
        mean = np.mean(total, axis=0)
        std = np.std(total, axis=0)
        std[std==0.0] = 1.0
        np.save(mean_std_filepath, [mean, std])
    print('process train')
    norm_train_fts = (train_fts - mean) / std
    np.save(os.path.join(feat_dir, 'train', 'speech_IS10_norm_ft.npy'), norm_train_fts)
    print('process validation')
    val_ft_filepath = os.path.join(feat_dir, 'val', 'speech_IS10_ft.npy')
    val_fts = np.load(val_ft_filepath, allow_pickle=True)
    norm_val_fts = (val_fts - mean) / std
    np.save(os.path.join(feat_dir, 'val', 'speech_IS10_norm_ft.npy'), norm_val_fts)
    print('process test')
    test_ft_filepath = os.path.join(feat_dir, 'test', 'speech_IS10_ft.npy')
    test_fts = np.load(test_ft_filepath, allow_pickle=True)
    norm_test_fts = (test_fts - mean) / std
    np.save(os.path.join(feat_dir, 'test', 'speech_IS10_norm_ft.npy'), norm_test_fts)

def get_movie_ind_norm_ft(feat_root_dir, output_root_dir):
    # 数据方便处理，将归一化的特征按照之前的格式保存起来
    movie2utt2ft = collections.OrderedDict()
    for setname in ['train', 'val', 'test']:
        set_ft_filepath = os.path.join(feat_root_dir, setname, 'speech_IS10_norm_ft.npy')
        set_int2name_filepath = os.path.join(feat_root_dir, setname, 'int2name.npy')
        set_fts = np.load(set_ft_filepath)
        set_int2name = np.load(set_int2name_filepath)
        assert len(set_fts) == len(set_int2name)
        pre_movie_name = set_int2name[0].split('_')[1]
        for uttId, ft in zip(set_int2name, set_fts):
            movie_name = uttId.split('_')[1]
            if pre_movie_name == movie_name:
                movie2utt2ft[uttId] = ft
            else:
                print('save {} and there {} utts'.format(pre_movie_name, len(movie2utt2ft)))
                save_filepath = os.path.join(output_root_dir, '{}_speech_ft_IS10_norm.pkl'.format(pre_movie_name))
                write_pkl(save_filepath, movie2utt2ft)
                movie2utt2ft = collections.OrderedDict()
                pre_movie_name = movie_name
                movie2utt2ft[uttId] = ft
        print('final movie is {} and {} utts'.format(movie_name, len(movie2utt2ft)))
        save_filepath = os.path.join(output_root_dir, '{}_speech_ft_IS10_norm.pkl'.format(movie_name))
        write_pkl(save_filepath, movie2utt2ft)

def get_wav2vec_finetuned_ft(set_ft_path, save_set_ft_path, feat_type, feat_root_dir, output_root_dir):
    # 数据方便处理，将归一化的特征按照之前的格式保存起来
    movie2utt2ft = collections.OrderedDict()
    for setname in ['train', 'val', 'test']:
        ori_set_ft_path = set_ft_path.format(setname)
        new_save_set_ft_path = save_set_ft_path.format(setname)
        print(ori_set_ft_path)
        print(new_save_set_ft_path)
        os.system('cp {} {}'.format(ori_set_ft_path, new_save_set_ft_path))
        set_int2name_filepath = os.path.join(feat_root_dir, setname, 'int2name.npy')
        set_fts = np.load(new_save_set_ft_path)
        set_int2name = np.load(set_int2name_filepath)
        assert len(set_fts) == len(set_int2name)
        pre_movie_name = set_int2name[0].split('_')[1]
        for uttId, ft in zip(set_int2name, set_fts):
            movie_name = uttId.split('_')[1]
            if pre_movie_name == movie_name:
                movie2utt2ft[uttId] = ft
            else:
                print('save {} and there {} utts'.format(pre_movie_name, len(movie2utt2ft)))
                save_filepath = os.path.join(output_root_dir, '{}_speech_ft_{}.pkl'.format(pre_movie_name, feat_type))
                # print(save_filepath)
                write_pkl(save_filepath, movie2utt2ft)
                movie2utt2ft = collections.OrderedDict()
                pre_movie_name = movie_name
                movie2utt2ft[uttId] = ft
        print('final movie is {} and {} utts'.format(movie_name, len(movie2utt2ft)))
        save_filepath = os.path.join(output_root_dir, '{}_speech_ft_{}.pkl'.format(movie_name, feat_type))
        # print(save_filepath)
        write_pkl(save_filepath, movie2utt2ft)

if __name__ == '__main__':
    output_dir = '/data9/memoconv/modality_fts/utt_baseline'
    split_info_dir = '/data9/MEmoConv/memoconv/split_set'
    target_dir = '/data9/memoconv/modality_fts/target/movies'

    if False:
        # Step1: 获取label的划分信息，以label信息为准，获取对应的特征数据（因为有些影响对话的句子可能删掉了）
        for setname in ['train', 'val', 'test']:
            print('current setname {}'.format(setname))
            set_movie_names_filepath = os.path.join(split_info_dir, '{}_movie_names.txt'.format(setname))
            int2name, int2label = get_set_label_info(set_movie_names_filepath, target_dir)
            print('utts {} check {} {}'.format(len(int2name), int2name[:5], int2label[:5]))
            output_label_filepath = os.path.join(output_dir, setname, 'label.npy')
            output_int2name_filepath = os.path.join(output_dir, setname, 'int2name.npy')
            np.save(output_label_filepath, int2label)
            np.save(output_int2name_filepath, int2name)
    
    if False:
        # Step2: 根据 int2name 获取对应的不同模态的特征, 注意统计长度，方便设计模型
        modality = 'visual' # text, speech, visual
        # 'bert_base_chinese'(768), 'robert_base_wwm_chinese'(768),, 'wav2vec_zh'(1024), 'comparE'(130) 'IS10'(1582) 'denseface'(342), 'lipresnet3d'(342)
        feature_type = 'affectdenseface'
        feature_dim = 342
        modality_ft_dir = os.path.join('/data9/memoconv/modality_fts/', modality, 'movies')
        for setname in ['train', 'val', 'test']:
            print('current setname {}'.format(setname))
            output_int2name_filepath = os.path.join(output_dir, setname, 'int2name.npy')
            set_movie_names_filepath = os.path.join(split_info_dir, '{}_movie_names.txt'.format(setname))
            set_fts = get_set_ft_info(set_movie_names_filepath, output_int2name_filepath, modality_ft_dir, modality, feature_type, feature_dim)
            np.save(os.path.join(output_dir, setname, '{}_{}_ft.npy'.format(modality, feature_type)), set_fts)
    
    if False:
        comparE_norm(output_dir)

    if False:
        IS10_norm(output_dir)

    if False:
        feat_root_dir = '/data9/memoconv/modality_fts/utt_baseline'
        output_root_dir = '/data9/memoconv/modality_fts/speech/movies'
        get_movie_ind_norm_ft(feat_root_dir, output_root_dir)

    if False:
        # get wav2vec finetuned features
        feat_type = 'sent_wav2vec_zh2chmed2e5last'
        setting_name = 'wav2vec_dnn_chmed_wav2vec_jonatasgrosman_wav2vec2-large-xlsr-53-chinese-zh-cn_2e-05'
        feat_root_dir = '/data9/memoconv/modality_fts/utt_baseline'
        output_root_dir = '/data9/memoconv/modality_fts/speech/movies'
        set_ft_path = os.path.join('/data9/MEmoConv/memoconv/results/utt_baseline/wav2vec_finetune', setting_name, 'epoch6_{}_ft.npy')
        save_set_ft_path = os.path.join(feat_root_dir, '{}/speech_'+ feat_type + '_ft.npy')
        get_wav2vec_finetuned_ft(set_ft_path, save_set_ft_path, feat_type, feat_root_dir, output_root_dir)