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

def get_latest_onlylva_result(path):
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
        if "tst_l_mask_av task" in line or "tst_l_mask_va task" in line:
            result_line = lines[index+3]
            WA, UAR = get_wa_ua_from_line(result_line)
            if UAR >= max_ua:
                max_ua = UAR
                max_ua_index = index
    best_step = get_above_best_step(max_ua_index, lines)
    print('best index {} step {}'.format(max_ua_index, best_step))
    # for tst_l_mask_av task
    current_setname_index =  max_ua_index
    result_line = lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['lva'] = [WA, UAR]
    if len(test_log) == 0:
        print('error of {}'.format(log_path))
    return test_log, best_step

def get_latest_onlypart_result(path, template='tst_l_mask'):
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
        if  template + " task" in line:
            result_line = lines[index+3]
            WA, UAR = get_wa_ua_from_line(result_line)
            if UAR >= max_ua:
                max_ua = UAR
                max_ua_index = index
    best_step = get_above_best_step(max_ua_index, lines)
    print('best index {} step {}'.format(max_ua_index, best_step))
    # for tst_l_mask task
    current_setname_index =  max_ua_index
    result_line = lines[current_setname_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['lva'] = [WA, UAR]
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

def get_all_6miss_cases_result(path):
    # 挑选6中miss cases的平均高UA最大的那组
    log_path = os.path.join(path, 'log.txt')
    f = open(log_path)
    lines = f.readlines()
    f.close()
    max_ua = 0
    max_ua_step = 0
    all_test_log = collections.OrderedDict()
    for index in range(len(lines)):
        line = lines[index]
        if 'Step' in line and 'start validation' in line:
            step_best_step = int(line[line.find('Step')+4: line.find(': start')])
            step_test_log = get_current_step_reuslt(index, lines)
            # get miss6 average ua 
            miss6_avg_wa = sum([step_test_log[sn][0] for sn in ['la', 'lv', 'l', 'va', 'a', 'v']]) / 6
            miss6_avg_ua = sum([step_test_log[sn][1] for sn in ['la', 'lv', 'l', 'va', 'a', 'v']]) / 6
            step_test_log['miss6_avg'] = [miss6_avg_wa, miss6_avg_ua]
            all_test_log[step_best_step] = step_test_log
            if miss6_avg_ua >= max_ua:
                max_ua = miss6_avg_ua
                max_ua_step = step_best_step
    print('best ua {} step {}'.format(max_ua, max_ua_step))
    return all_test_log[max_ua_step], max_ua_step

def get_current_step_reuslt(current_step_index, lines):
    test_log = collections.OrderedDict()
    lav_index = current_step_index + 1
    if 'val_l_mask_av task' in lines[lav_index] or 'val_l_mask_va task' in lines[lav_index]:
        # 如果包含val集合的话，那么过滤掉val部分的结果，只看test的结果
        lav_index = lav_index + 35
    assert 'tst_l_mask_av task' in lines[lav_index] or 'tst_l_mask_va task' in lines[lav_index]
    result_line = lines[lav_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['lva'] = [WA, UAR]
    # for tst_l_mask_v task
    lv_index = lav_index + 3 + 2
    assert 'tst_l_mask_v task' in lines[lv_index]
    result_line =  lines[lv_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['lv'] = [WA, UAR]
    # for tst_l_mask_a task
    la_index = lv_index + 3 + 2 
    assert 'tst_l_mask_a task' in lines[la_index]
    result_line =  lines[la_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['la'] = [WA, UAR]
    # for tst_l_mask task
    l_index = la_index + 3 + 2 
    assert 'tst_l_mask task' in lines[l_index]
    result_line =  lines[l_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['l'] = [WA, UAR]
    # for tst_mask_va task
    av_index = l_index + 3 + 2
    assert 'tst_mask_av task' in lines[av_index]
    result_line =  lines[av_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['va'] = [WA, UAR]
    # for tst_mask_v task
    v_index = av_index + 3 + 2
    assert 'tst_mask_v task' in lines[v_index]
    result_line =  lines[v_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['v'] = [WA, UAR]
    # for tst_mask_a task
    a_index = v_index + 3 + 2
    assert 'tst_mask_a task' in lines[a_index]
    result_line =  lines[a_index + 3]
    WA, UAR = get_wa_ua_from_line(result_line)
    test_log['a'] = [WA, UAR]
    return test_log

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
            all_lines.append('CV{}\t{:.2f}\t{:.2f}\n'.format(cvNo+1, wa_uas[cvNo][0], wa_uas[cvNo][1]))
            wa_results.append(wa_uas[cvNo][0])
            ua_results.append(wa_uas[cvNo][1])
        avg_wa = np.mean(wa_results)
        avg_ua = np.mean(ua_results)
        all_lines.append('Avg \t{:.2f}\t{:.2f}\n'.format(avg_wa, avg_ua))
    return all_lines

if __name__ == '__main__':
    root_dir = '/data7/emobert/exp/prompt_pretrain'
    output_name = 'iemocap-basedon-movies_v1v2v3_uniter3m_visual_wav2vec_text_5tasks_wwm_onespans.5v.5_noitm-cm_mask_prompt_onlylva_lr3e-5_trnval_part0.4_seed{}'
    type_eval = 'UA'
    for seed in [1234, 4321, 5678]:
        result_dir = os.path.join(root_dir, output_name.format(seed))
        # result_path = os.path.join(result_dir, 'result_miss6.csv')
        result_path = os.path.join(result_dir, 'result_lva.csv')
        all_tst_results = []
        for cvNo in range(1, 11):
            log_dir = os.path.join(result_dir, str(cvNo), 'log')
            if 'onlycm' in output_name:
                test_log, best_step = get_latest_onlycm_result(log_dir)
            elif 'nocm' in output_name:
                test_log, best_step = get_latest_nocm_result(log_dir)
            elif 'onlylva' in output_name or 'noaudio' in output_name or 'novisual' in output_name or '5tasks_noitm' in output_name:
                test_log, best_step = get_latest_onlylva_result(log_dir)
            elif 'only' in output_name:
                if 'onlylv' in output_name:
                    template = 'tst_l_mask_v'
                elif 'onlyl' in output_name:
                    template = 'tst_l_mask'
                elif 'onlyva' in output_name:
                    template = 'tst_mask_va'
                elif 'onlyv' in output_name:
                    template = 'tst_mask_v'
                elif 'onlya' in output_name:
                    template = 'tst_mask_a'
                test_log, best_step = get_latest_onlypart_result(log_dir, template)
            elif '7cases' in output_name:
                test_log, best_step = get_all_6miss_cases_result(log_dir)
            else:
                # test_log, best_step = get_all_6miss_cases_result(log_dir)
                test_log, best_step = get_latest_onlylva_result(log_dir)
            all_tst_results.append(test_log)
            # clean other ckpkts 
            ckpt_dir = os.path.join(result_dir, str(cvNo), 'ckpt')
            clean_other_ckpts(ckpt_dir, best_step)

        all_lines = get_final_results_format(all_tst_results)
        write_file(result_path, all_lines)