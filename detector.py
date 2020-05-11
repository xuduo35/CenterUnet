from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from models.network import create_model, load_model

try:
  from external.nms import soft_nms
except:
  print('NMS not imported! If you need it,'
        ' do \n cd external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from dataset.coco import COCO

class BaseDetector(object):
  def __init__(self, opt, output_dir):
    #if opt.gpus[0] >= 0:
    #  opt.device = torch.device('cuda')
    #else:
    #  opt.device = torch.device('cpu')

    self.output_dir = output_dir

    print('Creating model...')
    self.model = create_model(
            opt.network_type, opt.backbone,
            {'hm': opt.num_classes, 'wh': 2, 'reg': 2, 'allmask': opt.num_maskclasses*9}, opt.num_stacks,
            encoder_weights=None
            )
    self.model = load_model(self.model, opt.load_model)
    self.model = self.model.to(opt.device)
    self.model.eval()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = [1.0]
    self.opt = opt
    self.pause = True

  def pre_process(self, image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    if self.opt.fix_res:
      inp_height, inp_width = self.opt.input_h, self.opt.input_w
      c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
      s = max(height, width) * 1.0
    else:
      inp_height = (new_height | self.opt.pad) + 1
      inp_width = (new_width | self.opt.pad) + 1
      c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
      s = np.array([inp_width, inp_height], dtype=np.float32)

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    images = torch.from_numpy(images)
    meta = {
            'c': c, 's': s,
            'inp_height': inp_height,
            'inp_width': inp_width,
            'new_height': new_height,
            'new_width': new_width,
            'out_height': inp_height // self.opt.down_ratio,
            'out_width': inp_width // self.opt.down_ratio,
            'trans_inv': cv2.invertAffineTransform(trans_input)
            }
    return images, meta

  def process(self, images, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results):
   raise NotImplementedError

  def run(self, image_or_path_or_tensor, meta=None):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset='coco', ipynb=(self.opt.debug==3), theme=self.opt.debugger_theme)
    start_time = time.time()
    pre_processed = False
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''):
      image = cv2.imread(image_or_path_or_tensor)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True

    loaded_time = time.time()
    load_time += (loaded_time - start_time)

    detections = []
    for scale in self.scales:
      scale_start_time = time.time()
      if not pre_processed:
        images, meta = self.pre_process(image, scale, meta)
      else:
        # import pdb; pdb.set_trace()
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
      images = images.to(self.opt.device)

      if len(self.opt.gpus) > 0:
          torch.cuda.synchronize()

      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time

      output, dets, forward_time = self.process(images, return_time=True)

      if len(self.opt.gpus) > 0:
          torch.cuda.synchronize()

      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time

      if self.opt.debug >= 2:
        self.debug(debugger, images, dets, output, scale)

      dets = self.post_process(dets, meta, scale)

      if len(self.opt.gpus) > 0:
          torch.cuda.synchronize()

      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(dets)

    results = self.merge_outputs(detections)

    if len(self.opt.gpus) > 0:
        torch.cuda.synchronize()

    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time

    #if self.opt.debug >= 1:
    #  self.show_results(debugger, image, results)

    allmask = output['allmask'][0].detach().cpu().numpy().transpose(1,2,0)

    heatmap = output['hm'][0].detach().cpu().numpy().transpose(1,2,0)
    heatmap = np.amax(heatmap, axis=2)
    heatmap = (heatmap*255).astype(np.uint8)

    return {
            'results': results, 'meta': meta, 'allmask': allmask, 'heatmap': heatmap,
            'tot': tot_time, 'load': load_time,
            'pre': pre_time, 'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time
            }

class CtdetDetector(BaseDetector):
  def __init__(self, opt, output_dir):
    super(CtdetDetector, self).__init__(opt, output_dir)

  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]
      hm = output['hm']
      wh = output['wh']
      reg = output['reg'] if self.opt.reg_offset else None

      if len(self.opt.gpus) > 0:
          torch.cuda.synchronize()

      forward_time = time.time()
      dets = ctdet_decode(hm, wh, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)

    if return_time:
      return output, dets, forward_time
    else:
      return output, dets

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]

  def merge_outputs(self, detections):
    results = {}
    for j in range(1, self.num_classes + 1):
      results[j] = np.concatenate(
        [detection[j] for detection in detections], axis=0).astype(np.float32)
      if len(self.scales) > 1 or self.opt.nms:
         soft_nms(results[j], Nt=0.5, method=2)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
      kth = len(scores) - self.max_per_image
      thresh = np.partition(scores, kth)[kth]
      for j in range(1, self.num_classes + 1):
        keep_inds = (results[j][:, 4] >= thresh)
        results[j] = results[j][keep_inds]
    return results

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      #pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      #debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      #####
      heatmaps = output['hm'][i].detach().cpu().numpy().transpose(1,2,0)
      heatmaps = np.amax(heatmaps, axis=2)
      #heatmaps = (heatmaps > heatmaps.max()*0.01).astype(np.float32)
      pos = np.unravel_index(np.argmax(heatmaps), heatmaps.shape)
      print("max of heatmaps", heatmaps.max(), heatmaps.shape, pos, heatmaps[pos[0],pos[1]])
      heatmaps = (heatmaps*255).astype(np.uint8)
      # cv2.imwrite("./results/"+'pred_hm_{:.1f}'.format(scale)+".png", heatmaps)
      debugger.add_blend_img(img, heatmaps, 'pred_hm_{:.1f}'.format(scale))
      #####
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4],
                                 img_id='out_pred_{:.1f}'.format(scale))

    #debugger.save_img('pred_hm_{:.1f}'.format(scale), self.output_dir)
    debugger.save_img('out_pred_{:.1f}'.format(scale), self.output_dir)

  def show_results(self, debugger, image, results):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_classes + 1):
      for bbox in results[j]:
        if bbox[4] > self.opt.vis_thresh:
          debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
    debugger.show_all_imgs(pause=self.pause)
