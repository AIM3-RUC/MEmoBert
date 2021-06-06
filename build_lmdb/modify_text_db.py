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

def read_txt_db(txt_db_dir):
    env = lmdb.open(txt_db_dir)
    txn = env.begin(buffers=True)
    return txn

def modify_emotype(version, setname):
    txt_db_dir = f'/data7/emobert/txt_db/movies_{version}_th0.5_emowords_sentiword_all_{setname}.db'
    output_txt_db_dir = f'/data7/emobert/txt_db/movies_{version}_th0.5_emowords_sentiword_emocls_all_{setname}.db'
    text2img_path = os.path.join(txt_db_dir, 'txt2img.json')

    all_text2img_path = '/data7/emobert/txt_db/movies_{version}_th0.5_emowords_sentiword_all.db/txt2img.json'
    all_targe_path = '/data7/emobert/txt_pseudo_label/movie_txt_pseudo_label_{version}.h5'
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
    #     add_img_frame_key(version, setname)
    
    ## for iemocap or msp data
    # corpus_name = 'msp'
    # imgId2target = get_weak_lable_list(corpus_name)
    # for cvNo in range(1, 13):
    #     for setname in ['trn', 'tst', 'val']:
    #         print(f'current cv {cvNo} and set {setname}')
    #         modify_emotype_downstream(corpus_name, cvNo, setname)

    if True:
        for setname in ['val3k', 'trn3k', 'trn']:
            modify_emotype_vox(setname)