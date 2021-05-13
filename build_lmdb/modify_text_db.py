'''
在已有的txtdb中添加情感标签的信息, 
'''
import os
import json
import h5py
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

version = 'v1' #  v1 v2 v3
setname = 'val3k' # trn or trn3k or val3k
txt_db_dir = f'/data7/MEmoBert/emobert/txt_db/movies_{version}_th0.5_emowords_sentiword_all_{setname}.db'
output_txt_db_dir = f'/data7/MEmoBert/emobert/txt_db/movies_{version}_th0.5_emowords_sentiword_emocls_all_{setname}.db'
text2img_path = os.path.join(txt_db_dir, 'txt2img.json')

all_text2img_path = '/data7/MEmoBert/emobert/txt_db/movies_{version}_th0.5_emowords_sentiword_all.db/txt2img.json'
all_targe_path = '/data7/MEmoBert/emobert/txt_pseudo_label/movie_txt_pseudo_label_{version}.h5'
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