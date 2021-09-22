# basic paths
dataset_name = 'IEMOCAP'
# dataset_name = 'MSP-IMPROV'
result_dir = '/data7/emobert/exp/evaluation/{}/results'.format(dataset_name)
ft_dir = '/data7/emobert/exp/evaluation/{}/feature'.format(dataset_name)
target_dir = '/data7/emobert/exp/evaluation/{}/target'.format(dataset_name)

model_cfg = {
    # basic info
    'model_name': 'early_fusion_multi',
    'dataset_mode': 'iemocap_original', # use the original denseface features
    'pretained_ft_type': 'denseface_openface_iemocap_mean_std_torch',
    # global training info
    'dropout_rate': 0.5,
    'modality':'VL',
    'mid_fusion_layers':'256,128',  # fusion layers
    'output_dim':4,
    'bn':False,
    'batch_size':128,
    # for learning rate
    'learning_rate':2e-4,
    'lr_policy':'linear',
    # for training
    'fix_lr_epoch': 20, # real fix_lr_epoch = fix_lr_epoch - warmup_epoch
    'max_epoch': 40,
    'patience': 10,
    'warmup_epoch':0,
    'warmup_decay':0.01, # warmup_learning_rate = warmup_decay * learning_rate
    'optim':'adam',
    'betas':[0.9, 0.98],
    'grad_norm': 5.0,
    # for different module initializers:  none / orthogonal / xavier / normal / kaiming
    'init_type': 'normal',
    # visual encoer info -- lstm
    'max_visual_tokens': 50,
    'v_input_size':342,
    'v_embd_method':'maxpool', # use last mean att max
    'v_hidden_size':128,  # rnn
    # visual3d encoder info resnet3d + lstm
    'v_ft_name':'openface_iemocap_raw_img112',
    'v3d_img_size': 112, 
    'v3d_input_size': 512,
    'v3d_embd_method':'maxpool',
    'v3d_hidden_size': 128,
    # audio encoer info -- lstm
    'max_acoustic_tokens': 50,
    'a_ft_name': '',
    'a_input_size':768,
    'a_embd_method':'maxpool', # use last mean att max
    'a_hidden_size':128,
    # text encoder info -- textcnn, bert_base=768, bert_large=1024
    'max_lexical_tokens': 22,
    'l_ft_name': 'bert',
    'l_input_size': 768,
    'l_hidden_size': 128,
}