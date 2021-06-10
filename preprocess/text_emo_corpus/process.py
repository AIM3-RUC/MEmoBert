import os
import re
import random
from preprocess.FileOps import read_json, read_csv, write_csv, read_file
# export PYTHONPATH=/data7/MEmoBert/

emo_list = ['neutral', 'joy', 'sadness', 'fear', 'anger', 'surprise', 'disgust']

def process_emoline(save_root_dir):
    if not os.path.exists(save_root_dir):
        os.makedirs(save_root_dir)
    train_save_path = os.path.join(save_root_dir, 'emoline_all_train.tsv')
    test_save_path = os.path.join(save_root_dir, 'emoline_all_test.tsv')
    emo_sents = {}
    emo_test_sents = {}
    all_data_instances = []
    all_test_data_instances = []
    corpus_root = '/data3/zjm/dataset/EmotionLines/Friends'
    for setname in ['train', 'dev', 'test']:
        filepath = os.path.join(corpus_root, 'friends_{}.json'.format(setname))
        data = read_json(filepath)
        print(f'{len(data)} {filepath}')
        for session in data:
            for instance in session:
                sentence = instance['utterance'].replace('\t', ' ').replace('\n', ' ')
                # 判断是否有合法字母
                contain_en = bool(re.search('[a-z]', sentence))
                if contain_en is False:
                    continue
                # 判断长度符合要求
                splits = sentence.split(' ')
                if len(splits) <= 1:
                    continue            
                emo = instance['emotion']
                # 判断情感类别符合要求
                if emo in emo_list:
                    if emo == 'joy':
                        emo = 'happy'
                else:
                    continue
                
                if setname == 'test':
                    if emo_test_sents.get(emo) is None:
                        emo_test_sents[emo] = [sentence]
                    else:
                        emo_test_sents[emo] += [sentence]
                    all_test_data_instances.append([emo, sentence])
                else:
                    if emo_sents.get(emo) is None:
                        emo_sents[emo] = [sentence]
                    else:
                        emo_sents[emo] += [sentence]
                    all_data_instances.append([emo, sentence])
    for emo in emo_sents.keys():
        print(f'train emo {emo} sents {len(emo_sents[emo])}')
    for emo in emo_test_sents.keys():
        print(f'test emo {emo} sents {len(emo_test_sents[emo])}')
    print(f'all trainval instance {len(all_data_instances)}')
    print(f'all test instance {len(all_test_data_instances)}')
    write_csv(train_save_path, all_data_instances, delimiter='\t')
    write_csv(test_save_path, all_test_data_instances, delimiter='\t')

def multi_label_select4xed(emotions):
    '''
    首先在最基本的情感里面选择，比如 neutral anger joy sadness
    '''
    if '1' in emotions and '7' in emotions:
        return '1'
    if '5' in emotions:
        return '5'
    return emotions[0]

def process_xed(save_root_dir):
    if not os.path.exists(save_root_dir):
        os.makedirs(save_root_dir)
    save_path = os.path.join(save_root_dir, 'xed_all.tsv')
    emo_sents = {}
    all_data_instances = []
    emo2idmap = {"anger":1, "anticipation":2, "disgust":3, "fear":4, "joy":5, "sadness":6, "surprise":7, "trust":8, "neutral":9}
    id2emomap = {v:k for k, v in emo2idmap.items()}
    corpus_root = '/data3/zjm/dataset/XED/'
    for filename in ['en-annotated.tsv', 'neu_en.txt']:
        filepath = os.path.join(corpus_root, filename)
        data = read_csv(filepath, delimiter='\t')
        print(f'{len(data)} {filepath}')
        for instance in data:
            if 'neu_en' not in filename:
                sentence, emoIdx = instance[0], instance[1]
            else:
                sentence, emoIdx = instance[1], instance[0]
            # 判断是否有合法字母
            contain_en = bool(re.search('[a-z]', sentence))
            if contain_en is False:
                continue
            # 判断长度符合要求
            splits = sentence.replace('\t', ' ').replace('\n', ' ').split(' ')
            if len(splits) <= 1:
                continue
            # 多标签的，每个句子有两个标签 
            emoIdx = multi_label_select4xed(emoIdx.split(', '))
            emo = id2emomap[int(emoIdx)]
            if emo in emo_list:
                if emo == 'joy':
                    emo = 'happy'
            else:
                continue
            if emo_sents.get(emo) is None:
                emo_sents[emo] = [sentence]
            else:
                emo_sents[emo] += [sentence]
            all_data_instances.append([emo, sentence])
    for emo in emo_sents.keys():
        print(f'emo {emo} sents {len(emo_sents[emo])}')
    print(f'all instance {len(all_data_instances)}')
    write_csv(save_path, all_data_instances, delimiter='\t')

def process_emorynlp(save_root_dir):
    if not os.path.exists(save_root_dir):
        os.makedirs(save_root_dir)
    train_save_path = os.path.join(save_root_dir, 'emorynlp_all_train.tsv')
    test_save_path = os.path.join(save_root_dir, 'emorynlp_all_test.tsv')
    emo_test_sents = {}
    emo_sents = {}
    all_data_instances = []
    all_test_data_instances = []
    emomap = {"Peaceful": 'peaceful', "Mad":'anger', "Scared":'fear', "Joyful":'joy', "Sad": 'sadness', "Powerful":'powerful',  "Neutral":'neutral'}
    corpus_root = '/data3/zjm/dataset/emorynlp/emotion-detection-{}.json'
    for filename in ['trn', 'dev', 'tst']:
        filepath = corpus_root.format(filename)
        data = read_json(filepath)
        print(f'{len(data)} {filepath}')
        episodes = data['episodes']
        for episode in episodes:
            for scene in episode['scenes']:
                for instance in scene['utterances']:
                    sentence, emo = instance['transcript'], instance['emotion']
                    # 判断是否有合法字母
                    contain_en = bool(re.search('[a-z]', sentence))
                    if contain_en is False:
                        continue
                    # 判断长度符合要求
                    splits = sentence.replace('\t', ' ').replace('\n', ' ').split(' ')
                    if len(splits) <= 1:
                        continue
                    # 多标签的，每个句子有两个标签，生成多个句子 
                    emo = emomap[emo]
                    if emo in emo_list:
                        if emo == 'joy':
                            emo = 'happy'
                    else:
                        continue
                    if filename == 'tst':
                        if emo_test_sents.get(emo) is None:
                            emo_test_sents[emo] = [sentence]
                        else:
                            emo_test_sents[emo] += [sentence]
                        all_test_data_instances.append([emo, sentence])
                    else:
                        if emo_sents.get(emo) is None:
                            emo_sents[emo] = [sentence]
                        else:
                            emo_sents[emo] += [sentence]
                        all_data_instances.append([emo, sentence])
    for emo in emo_sents.keys():
        print(f'train emo {emo} sents {len(emo_sents[emo])}')
    for emo in emo_test_sents.keys():
        print(f'test emo {emo} sents {len(emo_test_sents[emo])}')
    print(f'all trainval instance {len(all_data_instances)}')
    print(f'all test instance {len(all_test_data_instances)}')
    write_csv(train_save_path, all_data_instances, delimiter='\t')
    write_csv(test_save_path, all_test_data_instances, delimiter='\t')

def process_dailydialog(save_root_dir):
    if not os.path.exists(save_root_dir):
        os.makedirs(save_root_dir)
    train_save_path = os.path.join(save_root_dir, 'dailydialog_all_train.tsv')
    test_save_path = os.path.join(save_root_dir, 'dailydialog_all_test.tsv')
    emo_sents = {}
    emo_test_sents = {}
    all_data_instances = []
    all_test_data_instances = []
    emomap = {0: 'neutral', 1: 'anger', 2: 'disgust', 3: 'fear', 4: 'joy', 5: 'sadness', 6: 'surprise'}
    filepath = '/data3/zjm/dataset/dailydialog/{}/dialogues_{}.txt'
    emo_filepath = '/data3/zjm/dataset/dailydialog/{}/dialogues_emotion_{}.txt'
    for setname in ['train', 'validation', 'test']:
        lines = read_file(filepath.format(setname, setname))
        emo_lines = read_file(emo_filepath.format(setname, setname))
        assert len(lines) == len(emo_lines)
        print(f'{len(lines)} {setname}')
        for index in range(len(lines)):
            utters = lines[index].strip().split('__eou__')[:-1]
            emoIds = emo_lines[index].strip().split(' ')
            assert len(utters) == len(emoIds)
            for sentence, emoId in zip(utters, emoIds):
                # 判断是否有合法字母
                contain_en = bool(re.search('[a-z]', sentence))
                if contain_en is False:
                    continue
                # 判断长度符合要求
                splits = sentence.replace('\t', ' ').replace('\n', ' ').split(' ')
                if len(splits) <= 1:
                    continue 
                emo = emomap[int(emoId)]
                if emo in emo_list:
                    if emo == 'joy':
                        emo = 'happy'
                else:
                    continue
                if setname == 'test':
                    if emo_test_sents.get(emo) is None:
                        emo_test_sents[emo] = [sentence]
                    else:
                        emo_test_sents[emo] += [sentence]
                    all_test_data_instances.append([emo, sentence])
                else:
                    if emo_sents.get(emo) is None:
                        emo_sents[emo] = [sentence]
                    else:
                        emo_sents[emo] += [sentence]
                    all_data_instances.append([emo, sentence])
    for emo in emo_sents.keys():
        print(f'train emo {emo} sents {len(emo_sents[emo])}')
    for emo in emo_test_sents.keys():
        print(f'test emo {emo} sents {len(emo_test_sents[emo])}')
    print(f'all trainval instance {len(all_data_instances)}')
    print(f'all test instance {len(all_test_data_instances)}')
    write_csv(train_save_path, all_data_instances, delimiter='\t')
    write_csv(test_save_path, all_test_data_instances, delimiter='\t')

def process_goemotions(save_root_dir):
    if not os.path.exists(save_root_dir):
        os.makedirs(save_root_dir)
    train_save_path = os.path.join(save_root_dir, 'goemotions_all_train.tsv')
    test_save_path = os.path.join(save_root_dir, 'goemotions_all_test.tsv')
    emo_test_sents = {}
    emo_sents = {}
    all_test_data_instances = []
    all_data_instances = []
    fine2emo = {}
    emotions_path = '/data3/zjm/dataset/goemotions/processed_data/emotions.txt'
    emotions = read_file(emotions_path)
    id2fine = {i:emotions[i].strip() for i in range(len(emotions))}
    print(id2fine)
    emomap_path = '/data3/zjm/dataset/goemotions/processed_data/ekman_mapping_strong.json'
    emomap = read_json(emomap_path)
    for emocat in emomap:
        fines = emomap[emocat]
        for f in fines:
            fine2emo[f] = emocat
    # add the neutral
    fine2emo['neutral'] = 'neutral'
    print(f'total {len(fine2emo)} fine emos belong to {len(emomap)}')
    filepath = '/data3/zjm/dataset/goemotions/processed_data/{}.tsv'
    for setname in ['train', 'dev', 'test']:
        data = read_csv(filepath.format(setname, setname), delimiter='\t')
        for instance in data:
            sentence, emoIdx = instance[0], instance[1]
            # 判断是否有合法字母
            contain_en = bool(re.search('[a-z]', sentence))
            if contain_en is False:
                continue
            # 判断长度符合要求
            splits = sentence.replace('\t', ' ').replace('\n', ' ').split(' ')
            if len(splits) <= 1:
                continue 
            #  多标签
            emoId = emoIdx.split(',')[0]
            fine_emo = id2fine[int(emoId)]
            if fine2emo.get(fine_emo) is None:
                continue
            emo = fine2emo[fine_emo]
            if emo in emo_list:
                if emo == 'joy':
                    emo = 'happy'
            else:
                continue
            if setname == 'test':
                if emo_test_sents.get(emo) is None:
                    emo_test_sents[emo] = [sentence]
                else:
                    emo_test_sents[emo] += [sentence]
                all_test_data_instances.append([emo, sentence])
            else:
                if emo_sents.get(emo) is None:
                    emo_sents[emo] = [sentence]
                else:
                    emo_sents[emo] += [sentence]
                all_data_instances.append([emo, sentence])
    for emo in emo_sents.keys():
        print(f'train emo {emo} sents {len(emo_sents[emo])}')
    for emo in emo_test_sents.keys():
        print(f'test emo {emo} sents {len(emo_test_sents[emo])}')
    print(f'all trainval instance {len(all_data_instances)}')
    print(f'all test instance {len(all_test_data_instances)}')
    write_csv(train_save_path, all_data_instances, delimiter='\t')
    write_csv(test_save_path, all_test_data_instances, delimiter='\t')

def process_all(root_dir, save_root_dir):
    all_trn_instances = []
    all_val_instances = []
    all_trn_emo2ins = {}
    all_val_emo2ins = {}
    for filename in ['emolines', 'xed', 'emorynlp', 'dailydialog', 'goemotions']:
        train_filepath = root_dir + '/' + filename + '/{}_all_train.tsv'.format(filename)
        test_filepath =  root_dir + '/' + filename + '/{}_all_test.tsv'.format(filename)
        if os.path.exists(train_filepath):
            train_filepath = train_filepath
        else:
            train_filepath = os.path.join(root_dir, filename, '{}_all.tsv'.format(filename))
        data = read_csv(train_filepath, delimiter='\t')
        for ins in data:
            emo, sent = ins[0], ins[1]
            all_trn_instances.append([emo, sent])
            if all_trn_emo2ins.get(emo) is None:
                all_trn_emo2ins[emo] = [sent]
            else:
                all_trn_emo2ins[emo] += [sent]
        if os.path.exists(test_filepath):
            data = read_csv(test_filepath, delimiter='\t')
            for ins in data:
                emo, sent = ins[0], ins[1]
                all_val_instances.append([emo, sent])
                if all_val_emo2ins.get(emo) is None:
                    all_val_emo2ins[emo] = [sent]
                else:
                    all_val_emo2ins[emo] += [sent]
    print('train {} val {}'.format(len(all_trn_instances), len(all_val_instances)))
    for emo in all_trn_emo2ins.keys():
        print(f'train emo {emo} sents {len(all_trn_emo2ins[emo])}')
    for emo in all_val_emo2ins.keys():
        print(f'test emo {emo} sents {len(all_val_emo2ins[emo])}')
    random.shuffle(all_trn_instances)
    random.shuffle(all_val_instances)
    write_csv(os.path.join(save_root_dir, 'train.tsv'),  all_trn_instances, delimiter='\t')
    write_csv(os.path.join(save_root_dir, 'val.tsv'),  all_val_instances, delimiter='\t')

def process_emo7_format(root_dir, save_root_dir):
    if not os.path.exists(save_root_dir):
        os.makedirs(save_root_dir)
    emo_list = {0: 'neutral', 1: 'happy', 2: 'surprise', 3: 'sadness', 4:'anger', 5: 'fear', 6: 'disgust'}
    emo2id = {v:k for k, v in emo_list.items()}
    print(emo2id)
    # add the head, label,sentence1 delimiter = ','
    for setname in ['train', 'val']:
        emo_count = {}
        new_all_instances = []
        new_all_instances.append(['label', 'sentence1'])
        filepath = os.path.join(root_dir, f'{setname}.tsv')
        all_instances = read_csv(filepath, delimiter='\t')
        for instance in all_instances:
            emo, sent = instance[0], instance[1]
            emo_idx = emo2id[emo]
            if emo_count.get(emo) is not None and emo_count.get(emo) > 20000:
                continue
            if emo_count.get(emo) is None:
                emo_count[emo] = 1
            else:
                emo_count[emo] += 1
            new_all_instances.append([emo_idx, sent])
        for emo in emo_count.keys():
            print(f'train emo {emo} sents {emo_count[emo]}')
        print('set {} {}'.format(setname, len(new_all_instances)))
        write_csv(os.path.join(save_root_dir, f'{setname}.csv'),  new_all_instances, delimiter=',')

def process_emo5_format(root_dir, save_root_dir):
    # add the head, label,sentence1 delimiter = ','
    # only use the neutral surprise, happy, sad, anger
    emo_list = {0: 'neutral', 1: 'happy', 2: 'surprise', 3: 'sadness', 4:'anger'}
    emo2id = {v:k for k, v in emo_list.items()}
    print(emo2id)
    # add the head, label,sentence1 delimiter = ','
    for setname in ['train', 'val']:
        emo_count = {}
        new_all_instances = []
        new_all_instances.append(['label', 'sentence1'])
        filepath = os.path.join(root_dir, f'{setname}.tsv')
        all_instances = read_csv(filepath, delimiter='\t')
        for instance in all_instances:
            emo, sent = instance[0], instance[1]
            if emo in ['fear', 'disgust']:
                continue
            if emo_count.get(emo) is not None and emo_count.get(emo) > 20000:
                continue
            if emo_count.get(emo) is None:
                emo_count[emo] = 1
            else:
                emo_count[emo] += 1
            emo_idx = emo2id[emo]
            new_all_instances.append([emo_idx, sent])
        for emo in emo_count.keys():
            print(f'train emo {emo} sents {emo_count[emo]}')
        print('set {} {}'.format(setname, len(new_all_instances)))
        write_csv(os.path.join(save_root_dir, f'{setname}.csv'),  new_all_instances, delimiter=',')

def process_emo4_format(root_dir, save_root_dir):
    # add the head, label,sentence1 delimiter = ','
    # only use the neutral surprise, happy, sad, anger
    emo_list = {0: 'anger', 1: 'happy', 2: 'neutral', 3: 'sadness'}
    emo2id = {v:k for k, v in emo_list.items()}
    print(emo2id)
    # add the head, label,sentence1 delimiter = ','
    for setname in ['train', 'val']:
        emo_count = {}
        new_all_instances = []
        new_all_instances.append(['label', 'sentence1'])
        filepath = os.path.join(root_dir, f'{setname}.tsv')
        all_instances = read_csv(filepath, delimiter='\t')
        for instance in all_instances:
            emo, sent = instance[0], instance[1]
            if emo in ['fear', 'disgust', 'surprise']:
                continue
            if emo_count.get(emo) is not None and emo_count.get(emo) > 20000:
                continue
            if emo_count.get(emo) is None:
                emo_count[emo] = 1
            else:
                emo_count[emo] += 1
            emo_idx = emo2id[emo]
            new_all_instances.append([emo_idx, sent])
        for emo in emo_count.keys():
            print(f'train emo {emo} sents {emo_count[emo]}')
        print('set {} {}'.format(setname, len(new_all_instances)))
        write_csv(os.path.join(save_root_dir, f'{setname}.csv'),  new_all_instances, delimiter=',')

if __name__ == "__main__":
    root_dir = '/data7/emobert/text_emo_corpus'

    if False:
        save_root_dir = os.path.join(root_dir, 'emolines')
        process_emoline(save_root_dir)

    if False:
        save_root_dir = os.path.join(root_dir, 'xed')
        process_xed(save_root_dir)
    
    if False:
        save_root_dir = os.path.join(root_dir, 'emorynlp')
        process_emorynlp(save_root_dir)
    
    if False:
        save_root_dir = os.path.join(root_dir, 'dailydialog')
        process_dailydialog(save_root_dir)
    
    if False:
        save_root_dir = os.path.join(root_dir, 'goemotions')
        process_goemotions(save_root_dir)
    
    if False:
        save_root_dir = os.path.join(root_dir, 'all_5corpus')
        process_all(root_dir, save_root_dir)
    
    if False:
        root_dir = os.path.join(root_dir, 'all_5corpus')
        save_root_dir = os.path.join(root_dir, 'emo7_bert_data')
        process_emo7_format(root_dir, save_root_dir)
    
    if False:
        root_dir = os.path.join(root_dir, 'all_5corpus')
        save_root_dir = os.path.join(root_dir, 'emo5_bert_data')
        process_emo5_format(root_dir, save_root_dir)

    if True:
        root_dir = os.path.join(root_dir, 'all_5corpus')
        save_root_dir = os.path.join(root_dir, 'emo4_bert_data')
        process_emo4_format(root_dir, save_root_dir)