import numpy as np
import sys
import os
import datetime
import collections
from glob import glob

def write_file(filepath, lines):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def get_latest_result(path, result_template):
    # 挑选UA最大的那组
    log_path = os.path.join(path, 'log.txt')
    f = open(log_path)
    lines = f.readlines()
    f.close()
    test_log = {}
    max_ua = 0
    for index in range(len(lines)):
        line = lines[index]
        if result_template in line:
            result_line = lines[index+3]
            # print(log_path)
            WA = float(result_line[result_line.find('WA:')+3: result_line.find('WF1:')].strip().replace(',', ''))
            UAR = float(result_line[result_line.find('UA:')+3:].strip().replace(',', ''))
            if UAR >= max_ua:
                test_log['UAR'] = UAR
                test_log['WA'] = WA
                max_ua = UAR
    if len(test_log) == 0:
        print('error of {}'.format(log_path))
    return test_log

if __name__ == '__main__':
    result_dir = '/data7/emobert/exp/prompt_pretrain'
    output_name = 'msp_basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm_step4w-cm_mask_prompt_lr5e-5_trnval'
    type_eval = 'UA'
    result_template = [
        'tst task',
        'tst_l_mask_av task',
        'tst_l_mask_v task',
        'tst_l_mask_a task',
        'tst_l_mask task',
        'tst_mask_av task',       
        'tst_mask_v task',       
        'tst_mask_a task',       
    ][7]
    result_dir = os.path.join(result_dir, output_name)
    result_path = os.path.join(result_dir, 'result_{}.csv'.format(result_template.replace(' ', '-')))
    all_lines = []
    all_tst_wa_results = []
    all_tst_ua_results = []
    for cvNo in range(1, 13):
        log_dir = os.path.join(result_dir, str(cvNo), 'log')
        test_log = get_latest_result(log_dir, result_template)
        all_tst_wa_results.append(test_log['WA'])
        all_tst_ua_results.append(test_log['UAR'])
        # remove one bad result and average 
        all_lines.append('CV{}\t{}\t{}\n'.format(cvNo, test_log['WA'], test_log['UAR']))
    avg_wa = np.mean(all_tst_wa_results)
    avg_ua = np.mean(all_tst_ua_results)
    all_lines.append('Avg\t{}\t{}\n'.format(avg_wa, avg_ua))
    write_file(result_path, all_lines)