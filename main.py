from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse

import torch

from utils.logger import Logger
from config import Config
from dataset.coco import COCO
from models.network import create_model, load_model, save_model
from trainer import CtdetTrainer
from utils.image import size2level, levelnum

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Object Detection!')

    parser.add_argument('--train_phase', default='end_to_end', help='train (pre_train_center/pre_train_box) phase')

    parser.add_argument('--without_gpu', action='store_true', default=False, help='no use gpu')

    parser.add_argument('--num_workers', type=int, default=4, help='dataloader threads. 0 for single-thread.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=140, help='total training epochs.')
    parser.add_argument('--save_all', action='store_true', help='save model to disk every 5 epochs.')
    parser.add_argument('--num_iters', type=int, default=-1, help='default: #samples / batch_size.')
    parser.add_argument('--val_intervals', type=int, default=5, help='number of epochs to run validation.')
    parser.add_argument('--trainval', action='store_true', help='include validation in training and test on test set')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for batch size 32.')
    parser.add_argument('--lr_step', type=str, default='90,120', help='drop learning rate by 10.')

    parser.add_argument('--sizeaug', action='store_true', default=False, help='size augmentation')

    parser.add_argument('--gpus', default='0', help='-1 for CPU, use comma for multiple gpus')
    parser.add_argument('--seed', type=int, default=326, help='random seed')

    parser.add_argument('--network_type', type=str, default='unetobj', help='network type')
    parser.add_argument('--backbone', type=str, default='peleenet', help='backbone network')

    parser.add_argument('--load_model', default='./save_models/model_last.pth', help='path to pretrained model')
    parser.add_argument('--resume', action='store_true', help='resume training')

    parser.add_argument('--test', action='store_true')

    parser.add_argument('--metric', default='loss', help='main metric to save best model')

    parser.add_argument('--convert_model', action='store_true', default=False, help='convert model')

    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    print(args)
    return args

def main():
  if not os.path.exists("./results"):
      os.mkdir("./results")

  args = get_args()

  torch.manual_seed(args.seed)

  args.gpus_str = args.gpus
  args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
  args.gpus = [i for i in range(len(args.gpus))] if args.gpus[0] >=0 else [-1]

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
  device = torch.device('cuda' if args.gpus[0] >= 0 else 'cpu')

  if args.network_type == 'large_hourglass':
      down_ratio = 4
      nstack = 2
  else:
      down_ratio = 2 if args.backbone == 'peleenet' else 1
      nstack = 1

  cfg = Config(
          args.gpus, device,
          args.network_type, args.backbone,
          down_ratio, nstack, train_phase=args.train_phase
          )

  if args.network_type != 'large_hourglass':
    if args.train_phase == 'pre_train_center':
      cfg.allmask_weight *= 0.1
      cfg.hm_weight *= 1.0
      cfg.wh_weight *= 0.1
      cfg.off_weight *= 0.1
    elif args.train_phase == 'pre_train_box':
      cfg.allmask_weight *= 1.0
      cfg.hm_weight *= 0.1
      cfg.wh_weight *= 1.0
      cfg.off_weight *= 1.0
    else:
      cfg.hm_weight *= 1.0
      cfg.wh_weight *= 1.0
      cfg.off_weight *= 1.0
      cfg.allmask_weight *= 1.0

  logger = Logger(cfg)

  cfg.update(COCO)
  dataset = COCO('train', cfg)

  if args.convert_model:
      model = create_model(cfg.network_type, args.backbone, {'hm': dataset.num_classes, 'wh': 2, 'reg': 2, 'allmask': dataset.num_maskclasses+levelnum}, cfg.num_stacks, encoder_weights=None)
      model, optimizer, start_epoch = load_model(model, cfg.load_model, True, resume=True)
      save_model(cfg.load_model.replace(".pth", "_"+cfg.network_type+".pth"), start_epoch, model)
      sys.exit(0)

  if args.verbose:
      dataset.verbose()
      return

  print('Creating model...')

  model = create_model(args.network_type, args.backbone, {'hm': dataset.num_classes, 'wh': 2, 'reg': 2, 'allmask': dataset.num_maskclasses+levelnum}, nstack)

  if args.network_type != 'large_hourglass':
    if args.train_phase == 'pre_train_center':
        optimizer = torch.optim.Adam([
          {'params': model.c_net.parameters(), 'lr': args.lr},
          {'params': model.b_net.parameters(), 'lr': args.lr*0.01},
          ])
    elif args.train_phase == 'pre_train_box':
        for para in model.c_net.parameters():
            para.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        optimizer = torch.optim.Adam([
          {'params': model.c_net.parameters(), 'lr': args.lr*0.01},
          {'params': model.b_net.parameters(), 'lr': args.lr},
          ])
  else:
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

  start_epoch = 0

  lr_step = [int(x) for x in args.lr_step.split(',')]

  if args.resume:
      model, optimizer, start_epoch = load_model(
              model,
              args.load_model if args.load_model != '' else cfg.load_model,
              optimizer,
              args.resume, args.lr,
              lr_step
              )

  trainer = CtdetTrainer(cfg, model, optimizer)
  trainer.set_device(cfg.gpus, cfg.chunk_sizes, cfg.device)

  print('Setting up data...')

  val_loader = torch.utils.data.DataLoader(
      COCO('val', cfg),
      batch_size=1,
      shuffle=False,
      num_workers=1,
      pin_memory=True
  )

  if args.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, cfg.save_dir)
    return

  train_loader = torch.utils.data.DataLoader(
      COCO('train', cfg),
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True,
      drop_last=True
  )

  print('Starting training...')

  best = 1e10

  for epoch in range(start_epoch + 1, args.num_epochs + 1):
    mark = epoch if args.save_all else 'last'

    log_dict_train, _ = trainer.train(epoch, train_loader)

    logger.write('epoch: {} |'.format(epoch))

    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))

    if args.val_intervals > 0 and epoch % args.val_intervals == 0:
      save_model(os.path.join(cfg.save_dir, 'model_{}.pth'.format(mark)), epoch, model, optimizer)

      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)

      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))

      if log_dict_val[args.metric] < best:
        best = log_dict_val[args.metric]

        save_model(os.path.join(cfg.save_dir, 'model_best.pth'), epoch, model)
    else:
      save_model(os.path.join(cfg.save_dir, 'model_last.pth'), epoch, model, optimizer)

    logger.write('\n')

    if epoch in lr_step:
      save_model(os.path.join(cfg.save_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)

      lr = args.lr * (0.1 ** (lr_step.index(epoch) + 1))

      print('Drop LR to', lr)

      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

  logger.close()

if __name__ == '__main__':
  main()
