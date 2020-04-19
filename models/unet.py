import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.heads import make_layer
from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.pspnet.decoder import PSPDecoder
from segmentation_models_pytorch.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.deeplabv3.decoder import DeepLabV3Decoder

def get_basenet(basenet, backbone, encoder_weights, classes, decoder_channels, activation='sigmoid'):
    if basenet == 'fpn':
        return smp.FPN(backbone, encoder_weights=encoder_weights, classes=classes, activation=activation)
    elif basenet == 'psp':
       return smp.PSPNet(backbone, encoder_weights=encoder_weights, classes=classes, activation=activation)
    elif basenet == 'deeplabv3':
        return smp.DeepLabV3(backbone, encoder_weights=encoder_weights, classes=classes, activation=activation)

    return smp.Unet(
            backbone,
            encoder_weights=encoder_weights,
            encoder_depth=len(decoder_channels),
            classes=classes,
            decoder_channels=decoder_channels,
            activation=activation
            )

def get_basenet_decoder(basenet, encoder_channels, decoder_channels):
    if basenet == 'fpn':
        return FPNDecoder(encoder_channels)
    elif basenet == 'psp':
        return PSPDecoder(encoder_channels)
    elif basenet == 'deeplabv3':
        return DeepLabV3Decoder(encoder_channels[-1], decoder_channels[-1])

    return UnetDecoder(
                encoder_channels=encoder_channels,
                decoder_channels=decoder_channels,
                n_blocks=len(decoder_channels),
                use_batchnorm=True,
                center=False,
                attention_type=None,
            )

class boxDecoder(nn.Module):
    def __init__(self, basenet, heads, nstack, interdim, encoder_channels, decoder_channels, input_channels=4):
        super(boxDecoder, self).__init__()

        self.nstack = nstack

        self.boxconv = nn.Conv2d(input_channels, 3, 3, 1, 1, bias=False)
        self.boxdecoder = get_basenet_decoder(
            basenet,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels
            )

        self.heads = heads
        self.generate_heads(interdim, decoder_channels)

        self.initialize()

    def generate_heads(self, interdim, decoder_channels):
        nstack = self.nstack
        heads = self.heads

        ## keypoint heatmaps
        for head in heads.keys():
          module = nn.ModuleList([
            make_layer(decoder_channels[-1], interdim, heads[head]) for _ in range(nstack)
          ])

          self.__setattr__(head, module)

    def initialize(self):
        ind = 0 # nstack is 1

        for head in self.heads:
            layer = self.__getattr__(head)[ind]
            self.initialize_module(layer)

    def initialize_module(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, c_net, input, out):
        boxconv = self.boxconv(input)

        boxencode = c_net.encoder(boxconv)
        boxdecode = self.boxdecoder(*boxencode)

        ind = 0 # nstack is 1

        for head in self.heads:
            layer = self.__getattr__(head)[ind]
            y = layer(boxdecode)

            out[head] = y

        return out

class UnetObj(nn.Module):
    def __init__(
        self,
        backbone = 'densenet169',
        heads = {'hm': 80, 'wh': 2, 'reg': 2},
        nstack = 1,
        num_classes = 80,
        in_channels = 3,
        decoder_channels = (256, 128, 64, 32, 16),
        encoder_weights = 'imagenet'
    ):
        super().__init__()

        assert(heads['hm'] == num_classes)

        if backbone == 'peleenet':
            decoder_channels = (64,32,16,8)

        self.basenet = "unet"
        self.heads = {}

        ## keypoint heatmaps
        for head in heads.keys():
          if 'hm' not in head:
            self.heads[head] = heads[head]

        self.c_net = get_basenet(self.basenet, backbone, encoder_weights, 80, decoder_channels)
        self.b_net = boxDecoder(
            self.basenet, self.heads, nstack, 256, self.c_net.encoder.out_channels, decoder_channels, 3+num_classes
            )

        self.initialize()

    def initialize_module(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def initialize(self):
        self.initialize_module(self.b_net)

    def forward(self, input):
        # raw result
        c_net_output = self.c_net(input)

        # concat
        csigmoid = F.sigmoid(c_net_output)
        csigmoidx2 = F.interpolate(csigmoid, (input.size()[2], input.size()[3]), mode='bilinear')
        b_net_input = torch.cat((input, csigmoidx2), 1)

        out = { 'hm': csigmoid }

        # second stage
        out = self.b_net(self.c_net, b_net_input, out)

        return [out]
