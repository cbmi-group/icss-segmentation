# -*- coding: utf-8 -*-
# File   : compute_params.py
# Author : Jiaxing Huang
# Email  : huangjiaxing2021@ia.ac.cn
# Date   : 21/07/2021
#
# This file is the function that computes the number of parameters and computation efficiencies of the models.
# https://github.com/cbmi-group/icss-segmentation
from __future__ import print_function

import os
import numpy as np
import torch

import datetime
import torch.nn as nn
from datasets.dataset import er_data_loader
from models.unet import UNet as u_net
from models.nested_unet import NestedUNet as u_net_plus
from models.deeplab_v3_modified import DeepLab as deeplab
from models.FCN import FCN32s, VGGNet,FCNs,FCN8s
from models.segnet_modified import SegNet
from models.agnet import AG_Net as ag_net
from models.hrnet import HRNetV2_origin,HRNetV2_modified
import cv2
import torch.nn.functional as F
from thop import profile
from thop import clever_format

print("PyTorch Version: ", torch.__version__)

'''
evaluation
'''


def eval_model(opts):
    val_batch_size = opts["eval_batch_size"]
    dataset_type = opts['dataset_type']
    load_epoch = opts['load_epoch']
    gpus = opts["gpu_list"].split(',')
    gpu_list = []
    for str_id in gpus:
        id = int(str_id)
        gpu_list.append(id)
    

    eval_data_dir = opts["eval_data_dir"]
    dataset_name = os.path.split(eval_data_dir)[-1].split('.')[0]

    train_dir = opts["train_dir"]
    model_type = opts['model_type']

    model_score_dir = os.path.join(str(os.path.split(train_dir)[0]),
                                   'predict_score/' + dataset_name + '_' + str(load_epoch))
    if not os.path.exists(model_score_dir): os.makedirs(model_score_dir)

    viz_dir = train_dir.replace('checkpoints', 'viz')
    seg_save = os.path.join(viz_dir, 'seg')
    p_seg_save = os.path.join(viz_dir, 'p_seg')
    if not os.path.exists(seg_save): os.makedirs(seg_save)
    if not os.path.exists(p_seg_save): os.makedirs(p_seg_save)

    # define network
    print("==> Create network")

    model = None

    if model_type == 'unet':
        model = u_net(1, 1)
    elif model_type == 'unetPlus':
        model = u_net_plus(1, 1)
    elif model_type == 'agnet':
        model = ag_net(2)
    elif model_type == 'penet':
        model = PE_Net()
    elif model_type == 'deeplab_origin':
        model = deeplab(backbone='resnet101', output_stride=16)
    elif model_type == 'deeplab_modified':
        model = deeplab(backbone='resnet101', output_stride=16)
    elif model_type == 'penet':
        model = PE_Net()
    elif model_type == 'hrnet_origin':
        model = HRNetV2_origin(1)
    elif model_type == 'hrnet_modified':
        model = HRNetV2_modified(1)
    elif model_type == 'segnet_modified':
        model = SegNet()
        model.init_weights()
    elif model_type == 'fcn32': 
        vgg_model = VGGNet(requires_grad=True, show_params=False)
        vgg_model.features[0]=nn.Conv2d(1, 64, kernel_size=3, padding=1)
        model = FCN32s(pretrained_net=vgg_model, n_class=1)
    elif model_type == 'fcn':
        print(1) 
        vgg_model = VGGNet(requires_grad=True, show_params=False)
        vgg_model.features[0]=nn.Conv2d(1, 64, kernel_size=3, padding=1)
        model = FCNs(pretrained_net=vgg_model, n_class=1)
    
    input1 = torch.randn(1, 1, 256, 256) 
    begin_time = datetime.datetime.now()
    for i in range(10):
      flops, params = profile(model, inputs=(input1,))
       
    end_time = datetime.datetime.now()
    d_time = end_time - begin_time
    
    print(flops/1e9,params/1e6) 
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params) 
    print(d_time.total_seconds())  

    

  


if __name__ == "__main__":
    opts = dict()
    opts['dataset_type'] = 'mito'

    opts["eval_batch_size"] = 3
    opts["gpu_list"] = "0,1,2,3"
    opts["train_dir"] = "./train_log/mito_train_unet_aug_v1_20210322_iouloss/checkpoints"
    opts["eval_data_dir"] = "./datasets/test_params.txt"

    # model_type = [unet, unetPlus, agnet, deeplab, penet]
    opts['model_type'] = 'segnet_modified'
    opts["load_epoch"] = 30

    eval_model(opts)


