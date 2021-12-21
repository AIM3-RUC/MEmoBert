import numpy as np
import sys
import os
import datetime
import collections
from glob import glob
from collections import OrderedDict

def write_file(filepath, lines):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def get_latest_lva_result(path, result_template):
    # 挑选UA最大的那组
    log_path = os.path.join(path, 'log.txt')
    f = open(log_path)
    lines = f.readlines()
    result_dict = None
    for index in range(len(lines)):
        line = lines[index]
        if result_template in line:
            order_index = line.index('OrderedDict')
            result_str = line[order_index:]
            result_dict = eval(result_str)
    if result_dict is None:
        print('error of {}'.format(log_path))
    return result_dict


if __name__ == '__main__':
    result_root = '/data7/MEmoBert/emobert/exp/evaluation/MSP/finetune'
    output_name = 'miss-nomask-movies-v1v2v3-base-uniter3m_speechwav2vec_5tasks_wwm_span_noitm_train4w_froze0-lr3e-5_trnval_trn6000_seed{}/drop0.1_frozen0_vqa_none'
    type_eval = 'UA'
    result_template = 'Test: OrderedDict'
    for seed in [1234, 4321]:
        all_lines = []
        all_tst_wa_results = []
        all_tst_ua_results = []
        result_dir = os.path.join(result_root, output_name.format(seed))
        result_path = os.path.join(result_dir,  'result_lva.csv')
        for cvNo in range(1, 13):
            log_dir = os.path.join(result_dir,  str(cvNo), 'log')
            test_log = get_latest_lva_result(log_dir, result_template)
            all_tst_wa_results.append(test_log['testlva']['WA'])
            all_tst_ua_results.append(test_log['testlva']['UA'])
            # remove one bad result and average 
            all_lines.append('CV{}\t{:.4f}\t{:.4f}\n'.format(cvNo, test_log['testlva']['WA'], test_log['testlva']['UA']))
        avg_wa = np.mean(all_tst_wa_results)
        avg_ua = np.mean(all_tst_ua_results)
        all_lines.append('Avg\t{:.4f}\t{:.4f}\n'.format(avg_wa, avg_ua))
        write_file(result_path, all_lines)