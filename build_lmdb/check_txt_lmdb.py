import lmdb
import msgpack
from lz4.frame import decompress

txt_db_dir = '/data2/ruc_vl_pretrain/vg/txt_db/pretrain_vg_val.db'
env = lmdb.open(txt_db_dir)
txn = env.begin(buffers=True)
item = msgpack.loads(decompress(txn.get('0'.encode('utf-8'))), raw=False)
print(item)

# conf_threshold 的值是否一致, 直接看 nbb_th0.0_max100_min10.json 的数值就行
filenames = ['No0026.Mrs.Doubtfire_367.npz',
            'No0077.Philadelphia_878.npz',
            'No0031.Any.Given.Sunday_1408.npz']
img_db_dir = '/data7/MEmoBert/emobert/img_db_nomask/movies_v1/'
db_name = 'nbb_th0.0_max100_min10'
env = lmdb.open(f'{img_db_dir}/{db_name}',
                             readonly=True, create=False)
txn = env.begin(buffers=True)
dump = txn.get('No0026.Mrs.Doubtfire_367.npz'.encode('utf-8'))
print(item)