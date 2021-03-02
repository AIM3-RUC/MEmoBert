data_dir = '/data3/zjm/dataset/ferplus/npy_data'
target_dir = '/data3/zjm/dataset/ferplus/npy_data'
result_dir = '/data7/MEmoBert/emobert/exp/face_model'

model_cfg = {
  'model_name': 'resnet18_adam_warmup_run2',
  'drop_rate': 0.0,
  'num_classes': 8,
  'block_type': 'basic', # basic or bottleneck
  'resblocks': [2, 2, 2, 2],
  # train_params as below 
  'batch_size': 64,
  'max_epoch': 100,
  'optimizer': 'adam',
  'nesterov': True, # for sgd
  'momentum': 0.9, # for sgd
  'weight_decay': 0, # for sgd
  'learning_rate': 0.0005,
  'warmup_epoch': 10,
  'warmup_decay': 0.001,  # 5e-7
  'fix_lr_epoch': 20, 
  'reduce_half_lr_epoch': 40,
  'reduce_half_lr_rate': 0.5,  # epochs * 0.5
  'patience': 15,
  'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
  'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
  'data_augmentation': True,
  'validation_set': True,
  'validation_split': None,
  'num_threads': 4,
  # for finetune
  "frozen_dense_blocks": 2,
}

#  resnet18: resblocks=[2, 2, 2, 2], block_type=basic
#  resnet34: resblocks=[3, 4, 6, 3], block_type=basic
#  resnet50: resblocks=[3, 4, 6, 3], block_type=bottle