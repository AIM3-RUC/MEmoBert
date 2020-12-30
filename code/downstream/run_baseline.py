import logging
import os, sys
from tqdm import tqdm
import numpy as np
import argparse
import torch
from torch.optim import lr_scheduler
from os.path import join
from code.downstream.configs import ef_config as config
from code.downstream.data import CustomDatasetDataLoader
from code.downstream.models.early_fusion_multi_model import EarlyFusionMultiModel
from code.downstream.utils.logger import get_logger
from code.uniter.optim.misc import build_optimizer
from code.uniter.utils.save import ModelSaver
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def clean_chekpoints(ckpt_dir, store_epoch):
    for checkpoint in os.listdir(ckpt_dir):
        if not checkpoint.startswith(str(store_epoch)+'_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(ckpt_dir, checkpoint))

def parse_with_config(parser):
    '''
    only update the model config, such as batch-size and dimension
    '''
    args = parser.parse_args()
    config_args = config.model_cfg
    override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                        if arg.startswith('--')}
    for k, v in config_args.items():
        if k not in override_keys:
            setattr(args, k, v)
    return args

def lambda_rule(epoch):
    assert opt.remain_epoch < opt.max_epoch
    niter = opt.remain_epoch
    niter_decay = opt.max_epoch - opt.remain_epoch
    lr_l = 1.0 - max(0, epoch + 1 - niter) / float(niter_decay + 1)
    return lr_l

def main(opt):
    ## for building the basic paths
    output_dir = join(config.result_dir, config.pretained_ft_type, config.model_name) # get logger path
        # for testing the parameters of the model
    setting_name = '{}_dp{}_bn{}_A{}_V{}_L{}_F{}_run{}'.format(opt.modality, opt.dropout_rate, \
        opt.bn, opt.a_hidden_size, opt.v_hidden_size, opt.l_hidden_size, \
        opt.mid_fusion_layers, opt.run_idx)
    
    output_dir = join(output_dir, setting_name, str(opt.cvNo))
    make_path(output_dir)
    log_dir = join(output_dir, 'log')
    checkpoint_dir = join(output_dir, 'ckpts')
    make_path(log_dir)
    make_path(checkpoint_dir)
    logger = get_logger(log_dir, 'none')
    logger.info('[Output] {}'.format(output_dir))

    ## build ft paths and target
    ft_dir = os.path.join(config.ft_dir, config.pretained_ft_type, str(opt.cvNo)) # setname + **.npy
    target_dir = ft_dir # save as setname + **.npy

    ## create a dataset given opt.dataset_mode and other options, the trn_db neither Dataset nor Dataloader
    trn_db = CustomDatasetDataLoader(opt, config.dataset_mode, ft_dir, target_dir, setname='trn', is_train=True)
    val_db = CustomDatasetDataLoader(opt, config.dataset_mode, ft_dir, target_dir, setname='val', is_train=False)
    tst_db = CustomDatasetDataLoader(opt, config.dataset_mode, ft_dir, target_dir, setname='tst', is_train=False)
    logger.info('The number of training samples = {}'.format(len(trn_db)))
    logger.info('The number of validation samples = {}'.format(len(val_db)))
    logger.info('The number of testing samples = {}'.format(len(tst_db)))

    model_saver = ModelSaver(checkpoint_dir)

    model = EarlyFusionMultiModel(opt)  # init model
    model.to(model.device)
    num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
    logger.info('[Model] parameters {}'.format(num_parameters))

    # Prepare model
    if opt.is_test and opt.restore_checkpoint:
        logger.info('[Model] At testing stage and restore from {}'.format(opt.restore_checkpoint))
        checkpoint = torch.load(opt.restore_checkpoint)
        model.load_state_dict(checkpoint)
    else:
        checkpoint = {}

    # initialized the optimizer
    optimizer = build_optimizer(model, opt)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    total_iters = 0                # the total number of training iterations
    best_eval_uar = 0              # record the best eval UAR
    best_eval_epoch = -1           # record the best eval epoch

    total_steps = opt.max_epoch * int((len(trn_db) / opt.batch_size))
    logger.info('Total iters {}'.format(total_steps))
    
    for epoch in tqdm(range(opt.max_epoch)):
        for i, batch in enumerate(trn_db):  # inner loop within one epoch
            total_iters += 1                # opt.batch_size
            model.set_input(batch)           # unpack data from dataset and apply preprocessing
            model.forward()   
            batch_loss = model.loss
            optimizer.zero_grad()  
            model.backward()            
            optimizer.step()
            # visual training loss every print_freq
            if total_iters % 20 == 0:   # print training losses and save logging information to the disk
                logger.info('Cur epoch {}'.format(epoch) + ' loss {}'.format(batch_loss))

        # for evaluation
        if epoch % 1 == 0:
            logger.info("============ Evaluation Epoch {} ============".format(epoch))
            logger.info("Cur learning rate {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
            val_log = evaluation(model, val_db)
            logger.info(f"[Validation] Loss: {val_log['loss']:.2f},"
                        f"\t WA: {val_log['WA']*100:.2f},"
                        f"\t UA: {val_log['UA']*100:.2f},\n")
            test_log = evaluation(model, tst_db)
            logger.info(f"[Testing] Loss: {test_log['loss']:.2f},"
                        f"\t WA: {test_log['WA']*100:.2f},"
                        f"\t UA: {test_log['UA']*100:.2f},\n")
            # update the current best model based on validation results
            if val_log['UA'] > best_eval_uar:
                best_eval_epoch = epoch
                best_eval_uar = val_log['UA']
                print('Save model at {} epoch'.format(epoch))
                model_saver.save(model, epoch)

        scheduler.step()

    # print best eval result
    logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_step_{}.pt'.format(best_eval_epoch))
    ck = torch.load(checkpoint_path)
    model.load_state_dict(ck)
    var_log = evaluation(model, val_db, save_dir=log_dir, set_name='val')
    logger.info('[Val] result acc %.4f uar %.4f f1 %.4f' % (var_log['WA'], var_log['UA'], var_log['F1']))
    tst_log = evaluation(model, tst_db, save_dir=log_dir, set_name='tst')
    logger.info('[Tst] result acc %.4f uar %.4f f1 %.4f' % (tst_log['WA'], tst_log['UA'], tst_log['F1']))
    # clean_chekpoints(checkpoint_dir, best_eval_epoch)

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
        cm = confusion_matrix(total_label, total_pred)
    except:
        acc, uar, f1, cm = 0,0,0,0
    model.train()
    # save test results
    if save_dir is not None:
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(set_name)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(set_name)), total_label)
    return {'loss': avg_loss,  'WA': acc,  'UA': uar, 'F1': f1, 'cm':cm}

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
    parser.add_argument('--cvNo', type=int, required=True,
                        help='which cross-valiation folder')
    parser.add_argument('--modality', type=str,
                        help='which modalities will consider, such as VL')

    # for stage
    parser.add_argument("--is_test", action='store_true')
    parser.add_argument("--restore_checkpoint", default=None, help='if at testing stage, then...')

    # for model
    parser.add_argument('--dropout_rate', type=float, help='')
    parser.add_argument('--mid_fusion_layers', type=str, default='256,128', help='')
    parser.add_argument('--lr', type=float, default='2e-4', help='learning rate')
    parser.add_argument('--bn', action='store_true', help='use bn for the fully connected layers')
    parser.add_argument('--a_hidden_size', type=int, default=128)
    parser.add_argument('--v_hidden_size', type=int, default=128)
    parser.add_argument('--l_hidden_size', type=int, default=128)

    parser.add_argument('--postfix', required=True, default='None',
                        help='postfix for the output dir')
    opt = parse_with_config(parser)
    main(opt)