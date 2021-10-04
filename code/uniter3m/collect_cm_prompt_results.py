import numpy as np
import sys
import os
import datetime
import collections
from glob import glob



'''
获取所有不同模态缺失场景下的结果, 可能存在也可能不存在
tst_l_mask_va task
tst_l_mask_v task
tst_l_mask_a task
tst_l_mask task
tst_mask_av task
tst_mask_v task
tst_mask_a task
'''

def write_file(filepath, lines):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def get_wa_ua_from_line(result_line):
    WA = float(result_line[result_line.find('WA:')+3: result_line.find('WF1:')].strip().replace(',', ''))
    UAR = float(result_line[result_line.find('UA:')+3:].strip().replace(',', ''))
    return WA, UAR

def get_latest_onlycm_result(path):
    # 挑选UA最大的那组
    log_path = os.path.join(path, 'log.txt')
    f = open(log_path)
    lines = f.readlines()
    f.close()
    test_log = collections.OrderedDict()
    max_ua = 0
    max_ua_index = 0
    for index in range(len(lines)):
        line = lines[index]
        if "tst_mask_av task" in line:
            result_line = lines[index+3]
            WA, UAR = get_wa_ua_from_line(result_line)
            if UAR >= max_ua:
                max_ua = UAR
                max_ua_index = index
    best_step = get_above_best_step(max_ua_index, lines)
    print('best index {} step {}'.format(max_ua_index, best_step))    # for tst_mask_va task
    current_setname_index = max_ua_index + 5*0
    assert 'tst_mask_av task' in lines[current_setname_index]
    result_line =  lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['va'] = [WA, UAR]
    # for tst_mask_v task
    current_setname_index = max_ua_index + 5*1
    assert 'tst_mask_v task' in lines[current_setname_index]
    result_line =  lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['v'] = [WA, UAR]
    # for tst_mask_a task
    current_setname_index = max_ua_index + 5*2
    assert 'tst_mask_a task' in lines[current_setname_index]
    result_line =  lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['a'] = [WA, UAR]
    if len(test_log) == 0:
        print('error of {}'.format(log_path))
    return test_log, best_step

def get_latest_nocm_result(path):
    # 挑选UA最大的那组
    log_path = os.path.join(path, 'log.txt')
    f = open(log_path)
    lines = f.readlines()
    f.close()
    test_log = collections.OrderedDict()
    max_ua = 0
    max_ua_index = 0
    for index in range(len(lines)):
        line = lines[index]
        if "tst_l_mask_va task" in line:
            result_line = lines[index+3]
            WA, UAR = get_wa_ua_from_line(result_line)
            if UAR >= max_ua:
                max_ua = UAR
                max_ua_index = index
    best_step = get_above_best_step(max_ua_index, lines)
    print('best index {} step {}'.format(max_ua_index, best_step))
    # for tst_l_mask_av task
    current_setname_index =  max_ua_index
    assert 'tst_l_mask_va task' in lines[current_setname_index]
    result_line = lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['lva'] = [WA, UAR]
    # for tst_l_mask_v task
    current_setname_index = max_ua_index + 5
    assert 'tst_l_mask_v task' in lines[current_setname_index]
    result_line =  lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['lv'] = [WA, UAR]
    # for tst_l_mask_a task
    current_setname_index = max_ua_index + 5*2
    assert 'tst_l_mask_a task' in lines[current_setname_index]
    result_line =  lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['la'] = [WA, UAR]
    # for tst_l_mask task
    current_setname_index = max_ua_index + 5*3
    assert 'tst_l_mask task' in lines[current_setname_index]
    result_line =  lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['l'] = [WA, UAR]
    if len(test_log) == 0:
        print('error of {}'.format(log_path))
    return test_log, best_step

def get_latest_seven_result(path):
    # 挑选UA最大的那组
    log_path = os.path.join(path, 'log.txt')
    f = open(log_path)
    lines = f.readlines()
    f.close()
    test_log = collections.OrderedDict()
    max_ua = 0
    max_ua_index = 0
    for index in range(len(lines)):
        line = lines[index]
        if "tst_l_mask_av task" in line:
            result_line = lines[index+3]
            WA, UAR = get_wa_ua_from_line(result_line)
            if UAR >= max_ua:
                max_ua = UAR
                max_ua_index = index
    best_step = get_above_best_step(max_ua_index, lines)
    print('best index {} step {}'.format(max_ua_index, best_step))
    # for tst_l_mask_av task
    current_setname_index =  max_ua_index
    assert 'tst_l_mask_av task' in lines[current_setname_index]
    result_line = lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['lva'] = [WA, UAR]
    # for tst_l_mask_v task
    current_setname_index = max_ua_index + 5
    assert 'tst_l_mask_v task' in lines[current_setname_index]
    result_line =  lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['lv'] = [WA, UAR]
    # for tst_l_mask_a task
    current_setname_index = max_ua_index + 5*2
    assert 'tst_l_mask_a task' in lines[current_setname_index]
    result_line =  lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['la'] = [WA, UAR]
    # for tst_l_mask task
    current_setname_index = max_ua_index + 5*3
    assert 'tst_l_mask task' in lines[current_setname_index]
    result_line =  lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['l'] = [WA, UAR]
    # for tst_mask_va task
    current_setname_index = max_ua_index + 5*4
    assert 'tst_mask_av task' in lines[current_setname_index]
    result_line =  lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['va'] = [WA, UAR]
    # for tst_mask_v task
    current_setname_index = max_ua_index + 5*5
    assert 'tst_mask_v task' in lines[current_setname_index]
    result_line =  lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['v'] = [WA, UAR]
    # for tst_mask_a task
    current_setname_index = max_ua_index + 5*6
    assert 'tst_mask_a task' in lines[current_setname_index]
    result_line =  lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['a'] = [WA, UAR]
    if len(test_log) == 0:
        print('error of {}'.format(log_path))
    return test_log, best_step

def get_above_best_step(max_ua_index, lines):
    best_step = 0
    best_line_index = 0
    for i in range(max_ua_index):
        line = lines[max_ua_index-i]
        if 'Step' in line and 'start validation' in line:
            best_step = int(line[line.find('Step')+4: line.find(': start')])
            best_line_index = i
            break
    assert best_line_index < 50
    return best_step

def clean_other_ckpts(ckpt_dir, store_epoch):
    # model_step_number.pt
    for checkpoint in os.listdir(ckpt_dir):
        if not checkpoint.endswith('_{}.pt'.format(store_epoch)):
            os.remove(os.path.join(ckpt_dir, checkpoint))

def get_final_results_format(all_tst_results):
    # 返回所有的结果 setting lva
    all_info = collections.OrderedDict()
    for cvNo in range(len(all_tst_results)):
        test_log = all_tst_results[cvNo]
        for key in test_log.keys():
            if all_info.get(key) is None:
                all_info[key] = [test_log[key]]
            else:
                all_info[key] += [test_log[key]]
    all_lines = []
    for setname in all_info.keys():
        all_lines.append(setname + '\n')
        wa_uas = all_info[setname]
        wa_results, ua_results = [], []
        for cvNo in range(len(wa_uas)):
            all_lines.append('CV{}\t{}\t{}\n'.format(cvNo+1, wa_uas[cvNo][0], wa_uas[cvNo][1]))
            wa_results.append(wa_uas[cvNo][0])
            ua_results.append(wa_uas[cvNo][1])
        avg_wa = np.mean(wa_results)
        avg_ua = np.mean(ua_results)
        all_lines.append('Avg \t{}\t{}\n'.format(avg_wa, avg_ua))
    return all_lines

if __name__ == '__main__':
    result_dir = '/data7/emobert/exp/prompt_pretrain'
    output_name = 'msp-basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_span_noitm-cm_mask_prompt_lr5e-5_trnval_part0.6_seed4321'
    type_eval = 'UA'
    result_dir = os.path.join(result_dir, output_name)
    result_path = os.path.join(result_dir, 'result.csv')
    all_tst_results = []
    for cvNo in range(1, 13):
        log_dir = os.path.join(result_dir, str(cvNo), 'log')
        if 'onlycm' in output_name:
            test_log, best_step = get_latest_onlycm_result(log_dir)
        elif 'nocm' in output_name:
            test_log, best_step = get_latest_nocm_result(log_dir)
        else:
            test_log, best_step = get_latest_seven_result(log_dir)
        all_tst_results.append(test_log)
        # clean other ckpkts 
        ckpt_dir = os.path.join(result_dir, str(cvNo), 'ckpt')
        clean_other_ckpts(ckpt_dir, best_step)

    all_lines = get_final_results_format(all_tst_results)
    write_file(result_path, all_lines)

# only cm 的中 va a v 的结果要比 seven 中 va a v 的结果要好.
# 可以适当的调节某些情况的比例