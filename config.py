import os

class Config():
  def __init__(
    self,
    gpus, device,
    network_type, backbone,
    batch_size, down_ratio, num_stacks=2, load_model='', train_phase='end_to_end'
    ):
    self.train_phase = train_phase

    self.gpus = gpus
    self.device = device

    self.data_dir = './data'
    self.save_dir = './save_models'
    self.debug_dir = os.path.join(self.save_dir, './debug')

    if load_model != '':
      self.load_model = load_model
    else:
      self.load_model = os.path.join(self.save_dir, 'model_{}.pth'.format('last'))

    self.keep_res = False

    self.pad = 32

    self.input_h = 384
    self.input_w = 384
    self.input_res = 384

    self.not_rand_crop = False
    self.scale = 0.4
    self.shift = 0.1
    self.flip = 0.5
    self.no_color_aug = False

    self.batch_size = batch_size
    self.down_ratio = down_ratio

    self.mse_loss = False
    self.reg_loss = 'l1'
    self.hm_gauss = -1 # xxx
    self.dense_wh = False
    self.norm_wh = False

    self.cat_spec_wh = False
    self.reg_offset = True

    self.sizeaug = True

    self.test = False
    self.debug = 1

    self.network_type = network_type
    self.backbone = backbone

    self.num_stacks = num_stacks

    self.eval_oracle_hm = False
    self.eval_oracle_wh = False
    self.eval_oracle_offset = False

    self.num_iters = -1

    self.task = 'ctdet'
    self.dataset = 'coco'

    self.allmask_weight = 1.0
    self.hm_weight = 1.0
    self.wh_weight = 1.0
    self.off_weight = 1.0

    self.hide_data_time = False
    self.print_iter = 1

    self.K = 100

    self.debugger_theme = 'white'
    self.center_thresh = 0.1

    self.chunk_sizes = [15]

    self.nms = False
    self.fix_res = True
    self.vis_thresh = 0.15

    # self.mean, self.std
    # self.num_classes
    # self.input_h, self.input_w
    # self.output_h, self.output_w

  def update(self, dataset):
    input_h, input_w = dataset.default_resolution
    self.mean, self.std = dataset.mean, dataset.std
    self.num_classes = dataset.num_classes
    self.num_maskclasses = 9

    # input_h(w): self.input_h overrides self.input_res overrides dataset default
    input_h = self.input_res if self.input_res > 0 else input_h
    input_w = self.input_res if self.input_res > 0 else input_w
    self.input_h = self.input_h if self.input_h > 0 else input_h
    self.input_w = self.input_w if self.input_w > 0 else input_w

    self.output_h = self.input_h // self.down_ratio
    self.output_w = self.input_w // self.down_ratio

    self.input_res = max(self.input_h, self.input_w)
    self.output_res = max(self.output_h, self.output_w)
