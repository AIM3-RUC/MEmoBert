# -*- coding:UTF-8 -*-
# https://github.com/wblgers/py_speech_seg
# 完善此接口，用于进行对话语音的分割。

from __future__ import print_function
import BIC.speech_segmentation as bic_seg

frame_size = 256
frame_shift = 128
sr = 16000

input_filepath = '/data8/hzp/tmp/output/syncnet/pyavi/clip1//audio.wav'
output_dir = '/data8/hzp/tmp/output/syncnet/pyavi/clip1//audio_save'
seg_point = bic_seg.multi_segmentation(input_filepath, sr, frame_size, frame_shift, plot_seg=False, save_seg=True,
                                   save_dir=output_dir, cluster_method='bic')
print('The segmentation point for this audio file is listed (Unit: /s)', seg_point)




