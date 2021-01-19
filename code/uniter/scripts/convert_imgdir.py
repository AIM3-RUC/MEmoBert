"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

convert image npz to LMDB
"""
import argparse
from ast import dump
import glob
import io
import json
import multiprocessing as mp
import os
from os.path import basename, exists
from token import NOTEQUAL

from cytoolz import curry
import numpy as np
from tqdm import tqdm
import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

def _compute_valid_nbb(img_dump, conf_th, max_bb):
    # 由于不同于bbx, 这里的 face 是有顺序的, 所以需要返回具体的index.
    if img_dump.get('confidence') is None:
        return range(len(img_dump['soft_labels']))
    else:
        valid_indexs = []
        for index in range(len(img_dump['confidence'])):
            if img_dump['confidence'][index] > conf_th:
                valid_indexs.append(index)
        if len(valid_indexs) > max_bb:
            valid_indexs = valid_indexs[:max_bb]
        return valid_indexs
        
@curry
def load_npz(conf_th, max_bb, fname):
    try:
        img_dump = np.load(fname, allow_pickle=True)
        valid_indexs = _compute_valid_nbb(img_dump, conf_th, max_bb)
        dump = {}
        for key, arr in img_dump.items():
            if arr.dtype == np.float32:
                arr = arr.astype(np.float16)
            if arr.ndim == 2:
                dump[key] = arr[valid_indexs, :]
            elif arr.ndim == 1:
                dump[key] = arr[valid_indexs]
            else:
                raise ValueError('wrong ndim')
    except Exception as e:
        # corrupted file
        print(f'corrupted file {fname}', e)
        dump = {}
        valid_indexs = []
    name = basename(fname)
    return name, dump, len(valid_indexs)


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


def dumps_msgpack(dump):
    return msgpack.dumps(dump, use_bin_type=True)


def main(opts):
    if opts.img_dir[-1] == '/':
        opts.img_dir = opts.img_dir[:-1]
    split = basename(opts.img_dir)
    db_name = (f'feat_th{opts.conf_th}_max{opts.max_bb}'
                    f'_min{opts.min_bb}')
    if opts.compress:
        db_name += '_compressed'
    if not exists(f'{opts.output}/{split}'):
        os.makedirs(f'{opts.output}/{split}')
    env = lmdb.open(f'{opts.output}/{split}/{db_name}', map_size=1024**4)
    txn = env.begin(write=True)
    files = glob.glob(f'{opts.img_dir}/*.npz')
    load = load_npz(opts.conf_th, opts.max_bb)
    name2nbb = {}
    with mp.Pool(opts.nproc) as pool, tqdm(total=len(files)) as pbar:
        for i, (fname, features, nbb) in enumerate(
                pool.imap_unordered(load, files, chunksize=128)):
            if not features:
                continue  # corrupted feature
            if opts.compress:
                dump = dumps_npz(features, compress=True)
            else:
                dump = dumps_msgpack(features)
            txn.put(key=fname.encode('utf-8'), value=dump)
            if i % 1000 == 0:
                txn.commit()
                txn = env.begin(write=True)
            name2nbb[fname] = nbb
            pbar.update(1)
        txn.put(key=b'__keys__',
                value=json.dumps(list(name2nbb.keys())).encode('utf-8'))
        txn.commit()
        env.close()
    with open(f'{opts.output}/{split}/'
                f'nbb_th{opts.conf_th}_'
                f'max{opts.max_bb}_min{opts.min_bb}.json', 'w') as f:
        json.dump(name2nbb, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default=None, type=str,
                        help="The input images.")
    parser.add_argument("--output", default=None, type=str,
                        help="output lmdb")
    parser.add_argument('--nproc', type=int, default=8,
                        help='number of cores used')
    parser.add_argument('--compress', action='store_true',
                        help='compress the tensors')
    parser.add_argument('--conf_th', type=float, default=0.0,
                        help='threshold for face detection confidence')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=100,
                        help='number of bounding boxes (fixed)')
    args = parser.parse_args()
    main(args)