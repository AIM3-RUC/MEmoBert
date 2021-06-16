'''
在已有的txtdb中添加情感标签的信息, 
'''
import os
import json
import h5py
import shutil
from cytoolz import curry
import numpy as np
from tqdm import tqdm
import lmdb
import msgpack
from lz4.frame import decompress
import msgpack_numpy
msgpack_numpy.patch()
from code.uniter.data.data import open_lmdb
from collections import defaultdict

def read_txt_db(txt_db_dir):
    env = lmdb.open(txt_db_dir)
    txn = env.begin(buffers=True)
    return txn

def modify_emotype(version, setname, postfix='5corpus_emo5'):
    txt_db_dir = f'/data7/emobert/txt_db/movies_{version}_th0.5_emowords_sentiword_all_{setname}.db'
    output_txt_db_dir = f'/data7/emobert/txt_db/movies_{version}_th0.5_emowords_sentiword_emocls_all_{setname}_{postfix}.db'
    text2img_path = os.path.join(txt_db_dir, 'txt2img.json')

    all_text2img_path = f'/data7/emobert/txt_db/movies_{version}_th0.5_emowords_sentiword_all.db/txt2img.json'
    all_targe_path = f'/data7/emobert/txt_pseudo_label/movie_txt_pseudo_label_{version}_all_{postfix}.h5'
    print(all_targe_path)
    all_textId2target = h5py.File(all_targe_path, 'r')
    all_text2img = json.load(open(all_text2img_path))
    print('total {} txts'.format(len(all_text2img)))
    assert len(all_textId2target.keys()) == len(all_text2img)

    # transfer to all imgId2target
    imgId2target = {}
    for textId in all_text2img.keys():
        img_fname = all_text2img[textId]
        target = all_textId2target[textId]
        imgId2target[img_fname] = target

    txn = read_txt_db(txt_db_dir)
    text2img = json.load(open(text2img_path))
    textIds = text2img.keys()
    open_db = curry(open_lmdb, output_txt_db_dir, readonly=False)
    with open_db() as db:
        for textId in tqdm(textIds, total=len(textIds)):
            example = msgpack.loads(decompress(txn.get(textId.encode('utf-8'))), raw=False)
            img_fname = example['img_fname']
            # get correct info by the img_fname
            emoinfo = imgId2target[img_fname]
            pred = emoinfo['pred'][0]
            logits = emoinfo['logits'][0]
            target = np.argmax(pred)
            assert example['id'] == textId
            example['soft_labels'] = np.array(pred)
            example['logits'] = np.array(logits)
            example['target'] = np.array(target)
            db[textId] = example
    os.system(f'cp {txt_db_dir}/*.json {output_txt_db_dir}/')

def modify_emotype_opensub(version):
    txt_db_dir = f'/data7/emobert/txt_db/onlytext_opensub_{version}_emo5_bert_data.db/'
    output_txt_db_dir = f'/data7/emobert/txt_db/onlytext_opensub_{version}_emo5_bert_data_5corpus_emo5_emocls.db/'
    text2img_path = os.path.join(txt_db_dir, 'txt2img.json')

    all_targe_path = f'/data7/emobert/txt_pseudo_label/onlytext_opensub_{version}_all_5corpus_emo5.h5'
    print(all_targe_path)
    all_textId2target = h5py.File(all_targe_path, 'r')
    text2img = json.load(open(text2img_path))
    textIds = text2img.keys()

    # transfer to all imgId2target
    imgId2target = {}
    for textId in text2img.keys():
        img_fname = text2img[textId]
        target = all_textId2target[textId]
        imgId2target[img_fname] = target

    txn = read_txt_db(txt_db_dir)
    open_db = curry(open_lmdb, output_txt_db_dir, readonly=False)
    with open_db() as db:
        for textId in tqdm(textIds, total=len(textIds)):
            example = msgpack.loads(decompress(txn.get(textId.encode('utf-8'))), raw=False)
            img_fname = example['img_fname']
            # get correct info by the img_fname
            emoinfo = imgId2target[img_fname]
            pred = emoinfo['pred'][0]
            logits = emoinfo['logits'][0]
            target = np.argmax(pred)
            assert example['id'] == textId
            example['soft_labels'] = np.array(pred)
            example['logits'] = np.array(logits)
            example['target'] = np.array(target)
            db[textId] = example
    os.system(f'cp {txt_db_dir}/*.json {output_txt_db_dir}/')

def get_high_quality_emo(version, setname):
    # 不光要保证质量，还要保证类别是均衡的。
    # bert-movies {'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3, 'anger': 4}
    txt_db_dir = f'/data7/emobert/txt_db/movies_{version}_th0.5_emowords_sentiword_all_{setname}.db'
    output_txt_db_dir = f'/data7/emobert/txt_db/movies_{version}_th0.5_emowords_sentiword_emoclsselected_all_{setname}.db'
    text2img_path = os.path.join(txt_db_dir, 'txt2img.json')

    all_text2img_path = f'/data7/emobert/txt_db/movies_{version}_th0.5_emowords_sentiword_all.db/txt2img.json'
    all_targe_path = f'/data7/emobert/txt_pseudo_label/movie_txt_pseudo_label_{version}.h5'
    all_textId2target = h5py.File(all_targe_path, 'r')
    all_text2img = json.load(open(all_text2img_path))
    assert len(all_textId2target.keys()) == len(all_text2img)

    # transfer to all imgId2target
    imgId2target = {}
    for textId in all_text2img.keys():
        img_fname = all_text2img[textId]
        target = all_textId2target[textId]
        imgId2target[img_fname] = target

    id2len = {}
    txt2img = {}  # not sure if useful
    img2txt = defaultdict(list)
    emo2count = {}
    txn = read_txt_db(txt_db_dir)
    text2img = json.load(open(text2img_path))
    textIds = text2img.keys()
    print('total {} txts'.format(len(text2img)))
    open_db = curry(open_lmdb, output_txt_db_dir, readonly=False)
    with open_db() as db:
        for textId in tqdm(textIds, total=len(textIds)):
            example = msgpack.loads(decompress(txn.get(textId.encode('utf-8'))), raw=False)
            img_fname = example['img_fname']
            # get correct info by the img_fname
            emoinfo = imgId2target[img_fname]
            pred = emoinfo['pred'][0]
            logits = emoinfo['logits'][0]
            target = np.argmax(pred)
            max_prob = pred[target]
            if len(example['input_ids']) <= 1:
                continue
            if target == 0:
                if max_prob >= 0.8:
                    pass
                else:
                    continue
            else:
                if max_prob >= 0.4:
                    pass
                else:
                    continue
            if emo2count.get(target) is None:
                emo2count[target] =  1
            else:
                emo2count[target] +=  1
            assert example['id'] == textId
            id2len[textId] = len(example['input_ids'])
            txt2img[textId] = img_fname
            img2txt[img_fname] = textId
            example['soft_labels'] = np.array(pred)
            example['logits'] = np.array(logits)
            example['target'] = np.array(target)
            db[textId] = example
    total_sampels = 0
    for emo in emo2count.keys():
        total_sampels += emo2count[emo]
        print(f'emo {emo} and count {emo2count[emo]}')
    print(f'high quality total {total_sampels} {len(id2len)} {len(txt2img)} {len(img2txt)}')
    with open(f'{output_txt_db_dir}/id2len.json', 'w') as f:
        json.dump(id2len, f)
    with open(f'{output_txt_db_dir}/txt2img.json', 'w') as f:
        json.dump(txt2img, f)
    with open(f'{output_txt_db_dir}/img2txts.json', 'w') as f:
        json.dump(img2txt, f)
    meta = {}
    meta['output'] = output_txt_db_dir
    meta['num_samples'] = total_sampels
    meta['UNK'] = 100
    meta['CLS'] = 101
    meta['SEP'] = 102
    meta['MASK'] = 103
    meta['v_range'] = [999, 30522]
    with open(f'{output_txt_db_dir}/meta.json', 'w') as f:
        json.dump(meta, f, indent=4)

def add_img_frame_key(version, setname):
    txt_db_dir = f'/data7/emobert/txt_db/movies_{version}_th0.5_emolare_all_{setname}.db'
    output_txt_db_dir = f'/data7/emobert/txt_db/movies_{version}_th0.5_emolare_all_{setname}.db_new'
    print(output_txt_db_dir)
    text2img_path = os.path.join(txt_db_dir, 'txt2img.json')
    txn = read_txt_db(txt_db_dir)
    text2img = json.load(open(text2img_path))
    textIds = text2img.keys()
    open_db = curry(open_lmdb, output_txt_db_dir, readonly=False)
    with open_db() as db:
        for textId in tqdm(textIds, total=len(textIds)):
            example = msgpack.loads(decompress(txn.get(textId.encode('utf-8'))), raw=False)
            img_fname = example['file_path']
            example['img_fname'] = img_fname
            db[textId] = example


def get_weak_lable_list(corpus_name):
    # for downsteam tasks
    imgId2target = {}
    for setname in ['trn', 'val', 'tst']:
        all_text2img_path = f'/data7/emobert/exp/evaluation/{corpus_name.upper()}/txt_db/1/{setname}_emowords_sentiword.db/txt2img.json'
        all_targe_path = f'/data7/emobert/txt_pseudo_label/{corpus_name}_txt_pseudo_label_{setname}_cv1.h5'
        all_textId2target = h5py.File(all_targe_path, 'r')
        all_text2img = json.load(open(all_text2img_path))
        assert len(all_textId2target.keys()) == len(all_text2img)
        # transfer to all imgId2target
        for textId in all_text2img.keys():
            img_fname = all_text2img[textId]
            target = all_textId2target[textId]
            imgId2target[img_fname] = target
    return imgId2target

def modify_emotype_downstream(corpus_name, cvNo, setname):
    # target 关键词是留给下游任务的真实标注的。
    txt_db_dir = f'/data7/emobert/exp/evaluation/{corpus_name.upper()}/txt_db/{cvNo}/{setname}_emowords_sentiword.db'
    txt_db_dir_new = f'/data7/emobert/exp/evaluation/{corpus_name.upper()}/txt_db/{cvNo}/{setname}_emowords_sentiword_emocls.db'
    shutil.copytree(txt_db_dir, txt_db_dir_new)
    text2img_path = os.path.join(txt_db_dir, 'txt2img.json')
    txn = read_txt_db(txt_db_dir)
    text2img = json.load(open(text2img_path))
    textIds = text2img.keys()
    open_db = curry(open_lmdb, txt_db_dir_new, readonly=False)
    with open_db() as db:
        for textId in tqdm(textIds, total=len(textIds)):
            example = msgpack.loads(decompress(txn.get(textId.encode('utf-8'))), raw=False)
            img_fname = example['img_fname']
            # get correct info by the img_fname
            emoinfo = imgId2target[img_fname]
            pred = emoinfo['pred'][0]
            logits = emoinfo['logits'][0]
            assert example['id'] == textId
            example['soft_labels'] = np.array(pred)
            example['logits'] = np.array(logits)
            db[textId] = example

def modify_emotype_vox(setname):
    txt_db_dir = f'/data7/emobert/txt_db/voxceleb2_v2_th1.0_emowords_sentiword_all_{setname}.db/'
    output_txt_db_dir = f'/data7/emobert/txt_db/voxceleb2_v2_th1.0_emowords_sentiword_emocls_all_{setname}.db/'
    text2img_path = os.path.join(txt_db_dir, 'txt2img.json')

    all_text2img_path = f'/data7/emobert/txt_db/voxceleb2_v2_th1.0_emowords_sentiword_all.db/txt2img.json'
    all_targe_path = '/data7/emobert/txt_pseudo_label/voxceleb2_txt_pseudo_label_v2.h5'
    all_textId2target = h5py.File(all_targe_path, 'r')
    all_text2img = json.load(open(all_text2img_path))
    print('total {} txts'.format(len(all_text2img)))
    assert len(all_textId2target.keys()) == len(all_text2img)

    # transfer to all imgId2target
    imgId2target = {}
    for textId in all_text2img.keys():
        img_fname = all_text2img[textId]
        target = all_textId2target[textId]
        imgId2target[img_fname] = target

    txn = read_txt_db(txt_db_dir)
    text2img = json.load(open(text2img_path))
    textIds = text2img.keys()
    open_db = curry(open_lmdb, output_txt_db_dir, readonly=False)
    with open_db() as db:
        for textId in tqdm(textIds, total=len(textIds)):
            example = msgpack.loads(decompress(txn.get(textId.encode('utf-8'))), raw=False)
            img_fname = example['img_fname']
            # get correct info by the img_fname
            emoinfo = imgId2target[img_fname]
            pred = emoinfo['pred'][0]
            logits = emoinfo['logits'][0]
            target = np.argmax(pred)
            assert example['id'] == textId
            example['soft_labels'] = np.array(pred)
            example['logits'] = np.array(logits)
            example['target'] = np.array(target)
            db[textId] = example

# export PYTHONPATH=/data7/MEmoBert
if __name__ == '__main__':
    ### for movies data
    # version = 'v3' #  v1 v2 v3
    # for setname in ['val3k', 'trn3k', 'trn']:
    #     for postfix in ['5corpus_emo5']:
    #         print(f'current {setname} {postfix}')
    #         modify_emotype(version, setname, postfix=postfix)

    ### for movies data
    # version = 'v3' #  v1 v2 v3
    # for setname in ['val3k', 'trn3k', 'trn']:
    #     add_img_frame_key(version, setname)

    ### for movies data, emotion selected
    version = 'v3' #  v1 v2 v3
    for setname in ['val3k', 'trn3k', 'trn']:
        get_high_quality_emo(version, setname)
    
    ## for iemocap or msp data
    # corpus_name = 'msp'
    # imgId2target = get_weak_lable_list(corpus_name)
    # for cvNo in range(1, 13):
    #     for setname in ['trn', 'tst', 'val']:
    #         print(f'current cv {cvNo} and set {setname}')
    #         modify_emotype_downstream(corpus_name, cvNo, setname)

    # if True:
    #     for setname in ['val3k', 'trn3k', 'trn']:
    #         modify_emotype_vox(setname)

    # if True:
    #     version = 'p4' #  v1 v2 v3
    #     modify_emotype_opensub(version)