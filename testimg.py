from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
import cv2
import numpy as np
import time
import torch

from external.nms import soft_nms

from utils.logger import Logger
from config import Config
from dataset.coco import COCO
from models.network import create_model, load_model, save_model
from detector import CtdetDetector as Detector

COCO_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
              'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
              'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
              'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
              'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
              'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
              'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Object Detection!')
    parser.add_argument('--finetuning', action='store_true', default=False, help='finetuning the training')
    parser.add_argument('--without_gpu', action='store_true', default=False, help='no use gpu')

    parser.add_argument('--num_workers', type=int, default=4, help='dataloader threads. 0 for single-thread.')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=140, help='total training epochs.')
    parser.add_argument('--save_all', action='store_true', help='save model to disk every 5 epochs.')
    parser.add_argument('--num_iters', type=int, default=-1, help='default: #samples / batch_size.')
    parser.add_argument('--val_intervals', type=int, default=5, help='number of epochs to run validation.')
    parser.add_argument('--trainval', action='store_true', help='include validation in training and test on test set')

    parser.add_argument('--lr', type=float, default=1.25e-4, help='learning rate for batch size 32.')
    parser.add_argument('--lr_step', type=str, default='90,120', help='drop learning rate by 10.')

    parser.add_argument('--sizeaug', action='store_true', default=False, help='size augmentation')

    parser.add_argument('--gpus', default='0', help='-1 for CPU, use comma for multiple gpus')
    parser.add_argument('--seed', type=int, default=326, help='random seed')

    parser.add_argument('--load_model', default='./save_models/model_last.pth', help='path to pretrained model')
    parser.add_argument('--resume', action='store_true', help='resume training')

    parser.add_argument('--test', action='store_true')

    parser.add_argument('--metric', default='loss', help='main metric to save best model')

    parser.add_argument('--image', default='./test.jpg', help='test image')
    parser.add_argument('--nms', action='store_true', default=False, help='nms')

    parser.add_argument('--network_type', type=str, default='large_hourglass', help='network type')
    parser.add_argument('--backbone', type=str, default='peleenet', help='backbone network')

    parser.add_argument('--output_dir', default='./results', help='output dir')
    parser.add_argument('--center_thresh', type=float, default=0.1, help='center threshold')

    args = parser.parse_args()
    print(args)
    return args

def test():
  args = get_args()

  args.gpus_str = args.gpus
  args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
  args.gpus = [i for i in range(len(args.gpus))] if args.gpus[0] >=0 else [-1]

  if not args.without_gpu:
      print("Use GPU")
      os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus_str
      device = torch.device('cuda')
  else:
      print("Use CPU")
      device = torch.device('cpu')
      args.gpus = []

  if args.network_type == 'large_hourglass':
      down_ratio = 4
      nstack = 2 
  else:
      down_ratio = 2 if args.backbone == 'peleenet' else 1
      nstack = 1

  cfg = Config(
          args.gpus, device,
          args.network_type, args.backbone,
          down_ratio, nstack
          )
  cfg.load_model = args.load_model
  cfg.nms = args.nms
  cfg.debug = 2
  cfg.center_thresh = args.center_thresh

  logger = Logger(cfg)

  cfg.update(COCO)
  dataset = COCO('val', cfg)

  # img = cv2.resize(img, (cfg.img_size,cfg.img_size), interpolation=cv2.INTER_CUBIC)

  # img = img.astype(np.float32) / 255.
  # img -= np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
  # img /= np.array(COCO_STD, dtype=np.float32)[None, None, :]

  # imgs = torch.FloatTensor(1, 3, cfg.img_size, cfg.img_size)

  # imgs[0,:,:,:] = torch.FloatTensor(img.transpose(2,0,1))
  # imgs = imgs.to(cfg.device)

  detector = Detector(cfg, args.output_dir)

  ret = detector.run(args.image)

  bbox_and_scores = ret['results']

  img = cv2.imread(args.image)
  h, w, _ = img.shape
  scale = 512/w
  w = 512
  h = int(h*scale)
  img = cv2.resize(img, (w,h))

  for key in bbox_and_scores:
  #  if key == 1:
      for box in bbox_and_scores[key]:
        if box[4] > cfg.center_thresh:
          x1 = int(box[0]*scale)
          y1 = int(box[1]*scale)
          x2 = int(box[2]*scale)
          y2 = int(box[3]*scale)
          print(x1, y1, x2, y2, COCO_NAMES[key], box[4])
          cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
          cv2.putText(img, COCO_NAMES[key], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (key*2, 255, 255-key), 2)

  cv2.imwrite(os.path.join(args.output_dir, "centernet.jpg"), img)

if __name__ == '__main__':
  test()
