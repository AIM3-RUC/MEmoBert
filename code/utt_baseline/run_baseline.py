import os, sys
import fcntl
import json
import numpy as np
import argparse
import torch
from torch.optim import lr_scheduler
from os.path import join
from code.utt_baseline.data import CustomDatasetDataLoader
from code.utt_baseline.models.early_fusion_multi_model import EarlyFusionMultiModel
from code.utt_baseline.utils.logger import get_logger
from code.utt_baseline.utils.save import ModelSaver
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clean_chekpoints(ckpt_dir, store_epochs):
    # model_step_number.pt
    for checkpoint in os.listdir(ckpt_dir):
        cur_epoch = int(checkpoint.split('_')[-1][:-3])
        if cur_epoch not in store_epochs:
            if 'model_step' in checkpoint:
                os.remove(os.path.join(ckpt_dir, checkpoint))

def parse_with_config(main_args):
    '''
    only update the model config, such as batch-size and dimension
    '''
    config_args = config.model_cfg
    override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                        if arg.startswith('--')}
    for k, v in config_args.items():
        if k not in override_keys:
            setattr(main_args, k, v)
    return main_args

def lambda_rule(epoch):
    '''
    比较复杂的策略, 返回的是学习率的衰减系数，而不是当前学习率
    在 warmup 阶段： 学习率保持很小的值，opt.learning_rate * opt.warmup_decay
    在 warmup < epoch <fix_lr_epoch 阶段，给定原始的学习率先训练几轮，10轮左右，经验值
    在 fix_lr_epoch < epoch 阶段，线性的进行学习率的降低
    '''
    if epoch < opt.warmup_epoch:
        return opt.warmup_decay
    else:
        assert opt.fix_lr_epoch < opt.max_epoch
        niter = opt.fix_lr_epoch
        niter_decay = opt.max_epoch - opt.fix_lr_epoch
        lr_l = 1.0 - max(0, epoch + 1 - niter) / float(niter_decay + 1)
        return lr_l

def main(opt):
    ## for building the basic paths
    output_dir = join(config.result_dir,  opt.model_name) # get logger path
        # for testing the parameters of the model
    setting_name = '{}_lr{}_dp{}_bn{}_A{}{}{}_V{}{}{}_L{}{}_F{}_run{}_{}'.format(opt.modality, opt.learning_rate, opt.dropout_rate, opt.bn, \
                                    opt.a_ft_type, opt.a_hidden_size, opt.a_embd_method, \
                                    opt.v_ft_type, opt.v_hidden_size, opt.v_embd_method, \
                                    opt.l_ft_type, opt.l_hidden_size, \
                                    opt.mid_fusion_layers, opt.run_idx, opt.postfix)
    output_config = join(output_dir, setting_name, 'config.json')
    output_tsv = join(output_dir, setting_name, 'result.tsv')
    if 'iemocap' in opt.dataset_mode or 'msp' in opt.dataset_mode:
        output_dir = join(output_dir, setting_name, str(opt.cvNo))
    else:
        output_dir = join(output_dir, setting_name)
    make_path(output_dir)

    with open(output_config, 'w') as f:
        optDict = opt.__dict__
        json.dump(optDict, f)

    if not os.path.exists(output_tsv):
        open(output_tsv, 'w').close()  # touch output_csv

    log_dir = join(output_dir, 'log')
    checkpoint_dir = join(output_dir, 'ckpts')
    make_path(log_dir)
    make_path(checkpoint_dir)
    logger = get_logger(log_dir, 'none')
    logger.info('[Output] {}'.format(output_dir))

    ft_dir = config.ft_dir
    target_dir = config.target_dir

    ## create a dataset given opt.dataset_mode and other options, the trn_db neither Dataset nor Dataloader
    trn_db = CustomDatasetDataLoader(opt, opt.dataset_mode, ft_dir, target_dir, 'trn,val', is_train=True)
    val_db = CustomDatasetDataLoader(opt, opt.dataset_mode, ft_dir, target_dir, 'tst', is_train=False)
    tst_db = CustomDatasetDataLoader(opt, opt.dataset_mode, ft_dir, target_dir, 'tst', is_train=False)
    logger.info('The number of training samples = {}'.format(len(trn_db)))
    logger.info('The number of validation samples = {}'.format(len(val_db)))
    logger.info('The number of testing samples = {}'.format(len(tst_db)))

    model_saver = ModelSaver(checkpoint_dir)

    model = EarlyFusionMultiModel(opt)  # init model
    model.to(model.device)
    num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
    logger.info('[Model] parameters {}'.format(num_parameters))
    print(model.parameters())

    # Prepare model
    if opt.is_test and opt.restore_checkpoint:
        logger.info('[Model] At testing stage and restore from {}'.format(opt.restore_checkpoint))
        checkpoint = torch.load(opt.restore_checkpoint)
        model.load_state_dict(checkpoint)
    else:
        checkpoint = {}

    # initialized the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate, betas=opt.betas)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    total_iters = 0                # the total number of training iterations
    best_eval_uar = 0              # record the best eval F1
    best_eval_uar_epoch = -1           # record the best eval F1 epoch
    patience = opt.patience

    total_steps = opt.max_epoch * int((len(trn_db) / opt.batch_size))
    logger.info('Total iters {}'.format(total_steps))
    
    for epoch in range(opt.max_epoch):
        for i, batch in enumerate(trn_db):  # inner loop within one epoch
            total_iters += 1                # opt.batch_size
            model.set_input(batch)           # unpack data from dataset and apply preprocessing
            model.forward()   
            batch_loss = model.loss
            optimizer.zero_grad()  
            model.backward()            
            optimizer.step()
            # visual training loss every print_freq
            if total_iters % 50 == 0:   # print training losses and save logging information to the disk
                logger.info(str('[Traing Loss:] {}'.format(batch_loss)))

        # for evaluation
        if epoch % 1 == 0:
            logger.info("============ Evaluation Epoch {} ============".format(epoch))
            logger.info("Cur learning rate {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
            val_log = evaluation(model, val_db)
            logger.info(str("[Validation] Loss: {:.2f}".format(val_log['loss']) +
                    "\t WA: {:.2f},".format(val_log['WA']*100) + 
                    "\t UAR: {:.2f},".format(val_log['UAR']*100) + 
                    "\t F1: {:.2f},".format(val_log['F1']*100) +
                    "\t WF1: {:.2f},".format(val_log['WF1']*100)))
            test_log = evaluation(model, tst_db)
            logger.info(str("[Testing] Loss: {:.2f}".format(test_log['loss']) + 
                    "\t WA: {:.2f},".format(test_log['WA']*100) + 
                    "\t UAR: {:.2f}".format(test_log['UAR']*100) + 
                    "\t F1: {:.2f}, ".format(test_log['F1']*100) + 
                    "\t WF1: {:.2f}, ".format(test_log['WF1']*100)))
            print('Save model at {} epoch'.format(epoch))
            model_saver.save(model, epoch)
            # update the current best model based on validation results
            if val_log['UAR'] > best_eval_uar:
                best_eval_uar_epoch = epoch
                best_eval_uar = val_log['UAR']
                # reset to init
                patience = opt.patience

        # for early stop
        if patience <= 0:            
            break
        else:
            patience -= 1
        # update the learning rate
        scheduler.step()

    # print best WF1 eval result
    logger.info('Loading best WF1 model found on val set: epoch-%d' % best_eval_uar_epoch)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_step_{}.pt'.format(best_eval_uar_epoch))
    if not os.path.exists(checkpoint_path):
        logger.error("Load checkpoint error, not exist such file")
        exit(0)
    ck = torch.load(checkpoint_path)
    model.load_state_dict(ck)
    val_log = evaluation(model, val_db, save_dir=log_dir, set_name='val')
    logger.info(str('[Val] WF1-result WA: %.4f UAR %.4f F1 %.4f WF1 %.4f' % (val_log['WA'], val_log['UAR'], val_log['F1'],  val_log['WF1'])))
    logger.info(str('\n{}'.format(val_log['cm'])))
    tst_log = evaluation(model, tst_db, save_dir=log_dir, set_name='test')
    logger.info(str('[Tst] WF1-result WA: %.4f UAR %.4f F1 %.4f WF1 %.4f' % (tst_log['WA'], tst_log['UAR'], tst_log['F1'], tst_log['WF1'])))
    logger.info(str('\n{}'.format(tst_log['cm'])))

    # print best F1 eval result
    logger.info('Loading best UAR model found on val set: epoch-%d' % best_eval_uar_epoch)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_step_{}.pt'.format(best_eval_uar_epoch))
    if not os.path.exists(checkpoint_path):
        logger.error("Load checkpoint error, not exist such file")
        exit(0)
    ck = torch.load(checkpoint_path)
    model.load_state_dict(ck)
    val_log = evaluation(model, val_db, save_dir=log_dir, set_name='val')
    logger.info(str('[Val] F1-result WA: %.4f UAR %.4f F1 %.4f WF1 %.4f' % (val_log['WA'], val_log['UAR'], val_log['F1'], val_log['WF1'])))
    logger.info(str('\n{}'.format(val_log['cm'])))
    tst_log = evaluation(model, tst_db, save_dir=log_dir, set_name='test')
    logger.info(str('[Tst] F1-result WA: %.4f UAR %.4f F1 %.4f WF1 %.4f' % (tst_log['WA'], tst_log['UAR'], tst_log['F1'], tst_log['WF1'])))
    logger.info(str('\n{}'.format(tst_log['cm'])))
    clean_chekpoints(checkpoint_dir, [best_eval_uar_epoch])
    write_result_to_tsv(output_tsv, tst_log, opt.cvNo)

def write_result_to_tsv(file_path, tst_log, cvNo):
    # 使用fcntl对文件加锁,避免多个不同进程同时操作同一个文件
    f_in = open(file_path)
    fcntl.flock(f_in.fileno(), fcntl.LOCK_EX) # 加锁
    content = f_in.readlines()
    if len(content) != 12:
        content += ['\n'] * (12-len(content))
    content[cvNo-1] = 'CV{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(cvNo, tst_log['WA'], tst_log['UAR'], tst_log['F1'])
    f_out = open(file_path, 'w')
    f_out.writelines(content)
    f_out.close()
    f_in.close() 

@torch.no_grad()
def evaluation(model, loader, set_name='val', save_dir=None):
    model.eval()
    total_pred = []
    total_target = []
    eval_loss = 0
    for i, batch in enumerate(loader):  # inner loop within one epoch
        model.set_input(batch)
        model.forward()
        eval_loss += model.loss
        # the predicton reuslts
        preds = model.pred.argmax(dim=1).detach().cpu().numpy()
        targets = batch['label'].detach().cpu().numpy()
        total_pred.append(preds)
        total_target.append(targets)
    avg_loss = eval_loss / len(total_pred)
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_target)
    try:
        # calculate metrics
        acc = accuracy_score(total_label, total_pred)
        uar = recall_score(total_label, total_pred, average='macro')
        f1 = f1_score(total_label, total_pred, average='macro')
        wf1 = f1_score(total_label, total_pred, average='weighted')
        cm = confusion_matrix(total_label, total_pred)
    except:
        acc, uar, f1, wf1, cm = 0,0,0,0, 0
    model.train()
    return {'loss': avg_loss,  'WA': acc,  'UAR': uar, 'F1': f1, 'WF1': wf1, 'cm':cm}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model for testing")
    parser.add_argument('--run_idx', type=int, required=True,
                        help='which multiple run')
    parser.add_argument('--gpu_id', type=int, required=True,
                        help='which gpu to run')
    parser.add_argument('--num_threads', type=int, default=1,
                        help='how many threads to use')
    parser.add_argument('--cvNo', type=int, default=1,
                        help='which cross-valiation folder')
    parser.add_argument('--modality', type=str,
                        help='which modalities will consider, such as VL')
    parser.add_argument('--dataset_mode', type=str,
                        help='which dataset will consider, such as chmed/iemcoap')
    parser.add_argument('--pretained_ft_type', type=str,
                        help='which feature will be use')
    
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--fix_lr_epoch', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--warmup_epoch', type=int, default=5)

    # for stage
    parser.add_argument("--is_test", action='store_true')
    parser.add_argument("--restore_checkpoint", default=None, help='if at testing stage, then...')

    # for model
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--dropout_rate', type=float, help='')
    parser.add_argument('--mid_fusion_layers', type=str, default='256,128', help='')
    parser.add_argument('--learning_rate', type=float, default='2e-4', help='learning rate')
    parser.add_argument('--bn', action='store_true', help='use bn for the fully connected layers')
    parser.add_argument('--a_ft_type', type=str, default=None)
    parser.add_argument('--v_ft_type', type=str, default=None)
    parser.add_argument('--l_ft_type', type=str, default=None)
    parser.add_argument('--a_hidden_size', type=int, default=128)
    parser.add_argument('--v_hidden_size', type=int, default=128)
    parser.add_argument('--l_hidden_size', type=int, default=128)
    parser.add_argument('--a_input_size', type=int, default=768)
    parser.add_argument('--v_input_size', type=int, default=342)
    parser.add_argument('--l_input_size', type=int, default=768)
    parser.add_argument('--max_text_tokens', type=int, default=20)
    parser.add_argument('--max_acoustic_tokens', type=int, default=128)
    parser.add_argument('--max_visual_tokens', type=int, default=64)
    parser.add_argument('--a_embd_method', type=str, default='maxpool')
    parser.add_argument('--v_embd_method', type=str, default='maxpool')

    parser.add_argument('--postfix', required=True, default='None',
                        help='postfix for the output dir')
    main_args = parser.parse_args()
    # 根据主函数传入的参数判断采用的config文件
    if 'chmed' in main_args.dataset_mode:
        print(' dataset chmed and Use chmed config ---- ')
        from configs import ef_chmed_config as config 
    elif 'iemocap_ori' in main_args.dataset_mode:
        from configs import ef_iemocap_ori_config as config 
    elif 'iemocap_pretrained' in main_args.dataset_mode:
        from configs import ef_iemocap_pretrained_config as config 
    elif 'msp_pretrained' in main_args.dataset_mode:
        from configs import ef_msp_pretrained_config as config 
    else:
        print('Error dataset_mode {}'.format(main_args.dataset_mode))

    opt = parse_with_config(main_args)
    main(opt)