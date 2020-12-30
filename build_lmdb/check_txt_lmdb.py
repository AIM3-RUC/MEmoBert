import lmdb
import msgpack
from lz4.frame import decompress

txt_db_dir = '/data2/ruc_vl_pretrain/vg/txt_db/pretrain_vg_val.db'
env = lmdb.open(txt_db_dir)
txn = env.begin(buffers=True)
item = msgpack.loads(decompress(txn.get('0'.encode('utf-8'))), raw=False)
print(item)

# 对比id和对应的vocab是否一致～