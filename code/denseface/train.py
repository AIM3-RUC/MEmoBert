import sys
import argparse
import os
import json
from os.path import join
import numpy as np
import torch
from torch.optim import lr_scheduler

import code.denseface.config.conf_fer as config
from code.denseface.data.fer import CustomDatasetDataLoader
from code.denseface.model.dense_net import DenseNet
from code.downstream.utils.logger import get_logger
from code.uniter.utils.save import ModelSaver
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

def main(opt):
    output_dir = join(config.result_dir, opt.model_name + '_{}{}_{}'.format(opt.optimizer, 
                                opt.learning_rate, opt.drop_rate))
    make_path(output_dir)

    output_config = join(output_dir, 'config.json')
    with open(output_config, 'w') as f:
        optDict = opt.__dict__
        json.dump(optDict, f)

    log_dir = join(output_dir, 'log')
    checkpoint_dir = join(output_dir, 'ckpts')
    make_path(log_dir)
    make_path(checkpoint_dir)
    logger = get_logger(log_dir, 'none')
    logger.info('[Output] {}'.format(output_dir))

    ## create a dataset given opt.dataset_mode and other options, the trn_db neither Dataset nor Dataloader
    trn_db = CustomDatasetDataLoader(opt, config.data_dir, config.target_dir, setname='trn', is_train=True)
    val_db = CustomDatasetDataLoader(opt, config.data_dir, config.target_dir, setname='val', is_train=False)
    tst_db = CustomDatasetDataLoader(opt, config.data_dir, config.target_dir, setname='tst', is_train=False)
    logger.info('The number of training samples = {}'.format(len(trn_db)))
    logger.info('The number of validation samples = {}'.format(len(val_db)))
    logger.info('The number of testing samples = {}'.format(len(tst_db)))

    model_saver = ModelSaver(checkpoint_dir)

    model = DenseNet(opt.gpu_id, growth_rate=opt.growth_rate, block_config=opt.block_config, 
                    num_init_features=opt.num_init_features, bn_size=opt.bn_size, 
                    compression_rate=opt.reduction, drop_rate=opt.drop_rate, 
                    num_classes=opt.num_classes)
    # to gpu card
    model.to(model.device)
    num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())
    logger.info('[Model] parameters {}'.format(num_parameters))
    # logger.info(model)

    # Prepare model
    if opt.is_test and opt.restore_checkpoint:
        logger.info('[Model] At testing stage and restore from {}'.format(opt.restore_checkpoint))
        checkpoint = torch.load(opt.restore_checkpoint)
        model.load_state_dict(checkpoint)
    else:
        checkpoint = {}

    # initialized the optimizer
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, 
                                momentum=opt.momentum, nesterov=opt.nesterov,
                                weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.reduce_half_lr_epoch, gamma=opt.reduce_half_lr_rate)

    best_eval_f1 = 0              # record the best eval UAR
    patience = opt.patience
    
    for epoch in range(opt.max_epoch):
        for i, batch in enumerate(trn_db):  # inner loop within one epoch
            model.set_input(batch)   
            model.forward()
            batch_loss = model.loss
            optimizer.zero_grad()  
            model.backward()            
            optimizer.step()
            if i % 100 == 0:
                logger.info('\t Cur train batch loss {}'.format(batch_loss))
        # for evaluation
        if epoch % 1 == 0:
            logger.info("============ Evaluation Epoch {} ============".format(epoch))
            logger.info("Cur learning rate {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
            val_log = evaluation(model, val_db)
            logger.info(f"[Validation] Loss: {val_log['loss']:.2f},"
                        f"\t F1: {val_log['F1']*100:.2f},"
                        f"\t WA: {val_log['WA']*100:.2f},"
                        f"\t UA: {val_log['UA']*100:.2f},\n")
            test_log = evaluation(model, tst_db)
            logger.info(f"[Testing] Loss: {test_log['loss']:.2f},"
                        f"\t F1: {test_log['F1']*100:.2f},"
                        f"\t WA: {test_log['WA']*100:.2f},"
                        f"\t UA: {test_log['UA']*100:.2f},\n")
            logger.info(test_log['cm'])
            logger.info('Save model at {} epoch'.format(epoch))
            model_saver.save(model, epoch)
            # update the current best model based on validation results
            if val_log['F1'] > best_eval_f1:
                best_eval_epoch = epoch
                best_eval_f1 = val_log['F1']
                # reset to init
                patience = opt.patience
        # for early stop
        if patience <= 0:            
            break
        else:
            patience -= 1
        # update the learning rate
        scheduler.step()

    # print best eval result
    logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_step_{}.pt'.format(best_eval_epoch))
    if not os.path.exists(checkpoint_path):
        logger.error("Load checkpoint error, not exist such file")
        exit(0)
    ck = torch.load(checkpoint_path)
    model.load_state_dict(ck)
    val_log = evaluation(model, val_db, save_dir=log_dir, set_name='val')
    logger.info('[Val] result WA: %.4f UAR %.4f F1 %.4f' % (val_log['WA'], val_log['UA'], val_log['F1']))
    logger.info('\n{}'.format(val_log['cm']))
    tst_log = evaluation(model, tst_db, save_dir=log_dir, set_name='tst')
    logger.info('[Tst] result WA: %.4f UAR %.4f F1 %.4f' % (tst_log['WA'], tst_log['UA'], tst_log['F1']))
    logger.info('\n{}'.format(tst_log['cm']))

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
        targets = batch['labels'].detach().cpu().numpy()
        total_pred.append(preds)
        total_target.append(targets)
    avg_loss = eval_loss / len(total_pred)
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_target)
    # calculate metrics
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    f1 = f1_score(total_label, total_pred, average='macro')
    cm = confusion_matrix(total_label, total_pred)
    model.train()
    # save test results
    if save_dir is not None:
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(set_name)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(set_name)), total_label)
    return {'loss': avg_loss,  'WA': acc,  'UA': uar, 'F1': f1, 'cm':cm}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--is_test', action='store_true', help='Test model')
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model for testing")
    parser.add_argument("--config_file",
                        default=None, type=str,
                        help="config filepath")
    parser.add_argument('--gpu_id', type=int, required=True,
                        help='which gpu to run')
    parser.add_argument('--learning_rate', type=float, help='init learning rate')
    parser.add_argument('--drop_rate', type=float, help='dropout rate')
    main_args = parser.parse_args()
    # 根据主函数传入的参数判断采用的config文件
    opt = parse_with_config(main_args)
    main(opt)