# -*- coding: utf-8 -*-
# File   : inference_img.py
# Author : Jiaxing Huang
# Email  : huangjiaxing2021@ia.ac.cn
# Date   : 21/07/2021
#
# This file is the function that generates the prediction images by using the trained models.
# https://github.com/cbmi-group/icss-segmentation
# -*- coding:utf-8 -*-
from __future__ import print_function

import os
import numpy as np
import torch
import torch.nn as nn
from datasets.dataset import er_data_loader
from models.unet import UNet as u_net
from models.nested_unet import NestedUNet as u_net_plus
from models.deeplab_v3_modified import DeepLab as deeplab
from models.agnet import AG_Net as ag_net
from models.hrnet import HRNetV2_origin,HRNetV2_modified
#from models.segnet_modified import SegNet
from models.FCN import FCN32s, VGGNet,FCNs,FCN8s
from models.segnet import SegNet
import cv2
from datasets.metric import *
import torch.nn.functional as F

print("PyTorch Version: ",torch.__version__)


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
    os.environ['CUDA_VISIBLE_DEVICE'] = opts["gpu_list"]

    eval_data_dir = opts["eval_data_dir"]
    dataset_name = os.path.split(eval_data_dir)[-1].split('.')[0]

    train_dir = opts["train_dir"]
    model_type = opts['model_type']

    model_score_dir = os.path.join(str(os.path.split(train_dir)[0]), 'predict_score/' + dataset_name + '_' + str(load_epoch))
    if not os.path.exists(model_score_dir): os.makedirs(model_score_dir)

    viz_dir = train_dir.replace('checkpoints', 'viz')
    seg_save = os.path.join(viz_dir, 'seg')
    p_seg_save = os.path.join(viz_dir, 'p_seg')
    if not os.path.exists(seg_save): os.makedirs(seg_save)
    if not os.path.exists(p_seg_save): os.makedirs(p_seg_save)

    # dataloader
    print("==> Create dataloader")
    dataloader= er_data_loader(eval_data_dir, val_batch_size, dataset_type, is_train = False)

    # define network
    print("==> Create network")

    model = None

    if model_type == 'unet':
        model = u_net(1,1)
    elif model_type == 'unetPlus':
        model = u_net_plus(1,1)
    elif model_type == 'agnet':
        model = ag_net(2)
    elif model_type == 'hrnet_origin':
        model = HRNetV2_origin(1)
    elif model_type == 'hrnet_modified':
        model = HRNetV2_modified(1)
    elif model_type == 'deeplab':
        model = deeplab(backbone='resnet101', output_stride=16)
    elif model_type == 'penet':
        model = PE_Net()
    elif model_type == 'segnet_modified':
        model = SegNet()
        model.init_weights()      
    elif model_type == 'fcn32': 
        vgg_model = VGGNet(requires_grad=True, show_params=False)
        vgg_model.features[0]=nn.Conv2d(1, 64, kernel_size=3, padding=1)
        model = FCN32s(pretrained_net=vgg_model, n_class=1)
    elif model_type == 'fcn': 
        vgg_model = VGGNet(requires_grad=True, show_params=False)
        vgg_model.features[0]=nn.Conv2d(1, 64, kernel_size=3, padding=1)
        model = FCNs(pretrained_net=vgg_model, n_class=1)

    # load trained model
    #pretrain_model = os.path.join(train_dir, "checkpoints_" + str(load_epoch) + ".pth")
    pretrain_model = os.path.join(train_dir, "best.pth")

    if os.path.isfile(pretrain_model):
        c_checkpoint = torch.load(pretrain_model)

        model.load_state_dict(c_checkpoint["model_state_dict"])

        print("==> Loaded pretrianed model checkpoint '{}'.".format(pretrain_model))

    else:
        print("==> No trained model.")
        return 0

    # set model to gpu mode
    print("==> Set to GPU mode")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=gpu_list)


    # enable evaluation mode
    with torch.no_grad():
        model.eval()
        total_img = 0
        #thred=0.5
        total_iou = 0.0


        for inputs in dataloader:
            images = inputs["image"].cuda()
            labels = inputs['mask']

            img_name = inputs['ID']

            total_img += len(images)

            p_seg = 0
            # unet
            if model_type == 'unet':
                p_seg = model(images)
            elif model_type == 'unetPlus':
                p_seg = model(images)
                p_seg = p_seg[-1]
            elif model_type == 'penet':
                p_seg, p_seg_down = model(images)
            # agnet
            elif model_type == 'agnet':
                out, side_5, side_6, side_7, side_8 = model(images)
                p_seg = F.softmax(side_8, dim=1)
            # deeplab
            elif model_type == 'deeplab':
                p_seg = model(images)          
             # hrnet
            elif model_type == 'hrnet_origin':
                p_seg = model(images)
                p_seg=p_seg[0]
            elif model_type == 'hrnet_modified':
                p_seg = model(images)
                p_seg=p_seg[0]
            elif model_type == 'fcn':
                p_seg = model(images)            
            elif model_type == 'fcn32':
                p_seg = model(images)
            elif model_type == 'segnet_modified':
                p_seg = model(images)
            else:
                print("type error")

            outputs=p_seg
            preds = outputs
            #preds = outputs > thred
        
            preds = preds.cpu()
                              
          
            
            val_iou = IoU(preds, labels)
            total_iou += val_iou

            for i in range(len(images)):

                print('predict image: {}'.format(img_name[i]))

                if model_type == 'agnet':
                    np.save(os.path.join(model_score_dir, img_name[i].split('.')[0] + '.npy'), preds[i][1].cpu().numpy().astype(np.float32))
                    cv2.imwrite(os.path.join(model_score_dir, img_name[i]), preds[i][1].cpu().numpy().astype(np.float32))
                else:
                    np.save(os.path.join(model_score_dir, img_name[i].split('.')[0] + '.npy'), preds[i][0].cpu().numpy().astype(np.float32))
                    cv2.imwrite(os.path.join(model_score_dir, img_name[i].split('.')[0] + '.tif'), preds[i][0].cpu().numpy().astype(np.float32))

        epoch_iou = total_iou / total_img
        print("validation image number {},=====> Evaluation IOU: {:.4f};".format(total_img,epoch_iou))


if __name__ == "__main__":
    opts = dict()
    opts['dataset_type'] = 'nucleus'

    opts["eval_batch_size"] = 32
    opts["gpu_list"] = "0,1,2,3"
    opts["train_dir"] = "./train_log/nucleus_train_best_fcn_mofidied/checkpoints"
    opts["eval_data_dir"] = "./datasets/test_nucleus_1004.txt"

    # model_type = [unet, unetPlus, agnet, deeplab, penet]
    opts['model_type'] = 'fcn'
    opts["load_epoch"] = 20

    eval_model(opts)


