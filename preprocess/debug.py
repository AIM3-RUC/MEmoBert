import os
from tasks.vision import *


# fer_idx_to_class = ['neu', 'hap', 'sur', 'sad', 'ang', 'dis', 'fea', 'con']

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 0
extract_denseface = DensefaceExtractor(device=device)
# face_path = '/data7/MEmoBert/preprocess/data/faces/No0001.The.Shawshank.Redemption/51/51_aligned/frame_det_00_000012.bmp'
# face_path = '/data7/MEmoBert/preprocess/data/faces/No0001.The.Shawshank.Redemption/71/71_aligned/frame_det_00_000071.bmp'
# face_path = '/data7/MEmoBert/preprocess/data/faces/No0019.Midnight.Sun/50/50_aligned/frame_det_01_000025.bmp'
face_path = '/data7/MEmoBert/preprocess/data/faces/No0065.Sorry.We.Missed.You/30/30_aligned/frame_det_00_000006.bmp'
ft, pred = extract_denseface(face_path)
print(ft.shape)
print(pred)

