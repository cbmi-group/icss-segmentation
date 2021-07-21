# -*- coding: utf-8 -*-
# File   : inference_unet.py
# Author : Jiaxing Huang
# Email  : huangjiaxing2021@ia.ac.cn
# Date   : 21/07/2021
#
# This file is the function that evaluates the performance in datasets of UNet and modified-UNets.
# https://github.com/cbmi-group/icss-segmentation
from __future__ import print_function
import copy
import os
import numpy as np
import torch
import xlwt
from datasets.dataset import er_data_loader
from models.unet import UNet as u_net
from models.nested_unet import NestedUNet as u_net_plus
from models.deeplab_v3 import DeepLab as deeplab
from models.agnet import AG_Net as ag_net
from models.FCN import FCN32s, VGGNet,FCNs,FCN8s
from models.hrnet import HRNetV2_origin,HRNetV2_modified
from datasets.metric import *
import cv2
import torch.nn.functional as F

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
    os.environ['CUDA_VISIBLE_DEVICE'] = opts["gpu_list"]

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

    # dataloader
    print("==> Create dataloader")

    dataloader = er_data_loader(eval_data_dir, val_batch_size, dataset_type, is_train=False)

    # define network
    print("==> Create network")

    model = None

    if model_type == 'unet':
        model = u_net(1, 1)
    elif model_type == 'unetPlus':
        model = u_net_plus(1, 1)
    elif model_type == 'agnet':
        model = ag_net(2)
    elif model_type == 'deeplab_origin':
        model = deeplab(backbone='resnet50', output_stride=16)
    elif model_type == 'penet':
        model = PE_Net()
    elif model_type == 'hrnet_origin':
        model = HRNetV2_origin(1)
    elif model_type == 'hrnet_modified':
        model = HRNetV2_modified(1)
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
    #device = torch.device('cuda:1')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    wb = xlwt.Workbook(encoding='ascii')  

    ws = wb.add_sheet('wg')             

    ws.write(0, 0, label='Threshold')        

    ws.write(0, 1, label='IOU')        

    ws.write(0 ,2, label='ACC')
    ws.write(0 ,3, label='AUC')
    ws.write(0 ,4, label='F1 Score')

    

    # enable evaluation mode
    with torch.no_grad():
      model.eval()
      model_dict = model.state_dict()
      model_dict_copy = copy.deepcopy(model_dict)
     
      #model_dict_copy['module.up4.conv.double_conv.0.weight'][:,:64,:,:] = torch.zeros(64,64,3,3)
      #model_dict_copy['module.up3.conv.double_conv.0.weight'][:,:128,:,:] = torch.zeros(128,128,3,3)
      #model_dict_copy['module.up2.conv.double_conv.0.weight'][:,:256,:,:] = torch.zeros(256,256,3,3)
      #model_dict_copy['module.up1.conv.double_conv.0.weight'][:,:512,:,:] = torch.zeros(512,512,3,3)
      model.load_state_dict(model_dict_copy)
  
      
      for Threshold in range(50,70):
        
            
            
           
 
        # param_weights = model_dict_copy['up4.conv.double_conv.0.weight']
        # param_bias = model_dict_copy['up4.conv.double_conv.0.bias']
        # print(param_weights.size())
        # print(param_bias.size())

           
            
            #model_dict_copy['up2.conv.double_conv.0.weight'][:,:256,:,:] = torch.zeros(256,256,3,3)

            

            total_iou = 0.0
            total_f1 = 0.0
            total_distance = 0.0
            total_acc = 0.0
            total_img = 0
            total_auc = 0.0
            f_threshold=Threshold*0.01

            for inputs in dataloader:
                images = inputs["image"].cuda()
                labels = inputs['mask']
                auc_labels=labels
                auc_labels[labels>=1]=1
                auc_labels[labels<1]=0


                img_name = inputs['ID']

                total_img += len(images)

                
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
                elif model_type == 'deeplab_origin':
                    p_seg = model(images)
                    
                # deeplab
                elif model_type == 'deeplab_modified':
                    p_seg = model(images)
                
                 # hrnet
                elif model_type == 'hrnet_origin':
                    p_seg = model(images)
                    p_seg=p_seg[0]
                 # hrnet
                elif model_type == 'hrnet_modified':
                    p_seg = model(images)
                    p_seg=p_seg[0]
                # deeplab
                elif model_type == 'fcn':
                    p_seg = model(images)
                # deeplab
                elif model_type == 'fcn32':
                    p_seg = model(images)
                else:
                    p_seg = 0
                          
                outputs=p_seg
                preds = outputs > f_threshold
                
                preds = preds.cpu()
                outputs = outputs.cpu()

                # metric
                val_auc = roc_list(outputs, auc_labels)
                total_auc += val_auc

                val_acc = acc_list(preds, labels)
                total_acc += val_acc

                val_iou = IoU(preds, labels)
           
                total_iou += val_iou

                h_distance = hausdorff_distance(preds, labels)
                total_distance += h_distance

                val_f1 = F1_score(preds, labels)
                total_f1 += val_f1
                # iou
            epoch_iou = total_iou / total_img
            epoch_f1 = total_f1 / total_img
            epoch_distance = total_distance / total_img
            epoch_acc = total_acc / total_img
            epoch_auc = total_auc / total_img
            ws.write(Threshold, 0, f_threshold)        
            ws.write(Threshold, 1, format(float(epoch_iou), '.4f'))        
            ws.write(Threshold ,2, format(float(epoch_acc), '.4f'))
            ws.write(Threshold ,3, format(float(epoch_auc), '.4f'))
            ws.write(Threshold ,4,  format(float(epoch_f1), '.4f'))

            message = "total Threshold: {:.3f} =====> Evaluation IOU: {:.4f}; F1_score: {:.4f}; AUC: {:.4f}; ACC: {:.4f}".format(
                f_threshold, epoch_iou, epoch_f1, epoch_auc, epoch_acc)
            print("==> %s" % (message))
    wb.save(model_type+dataset_type+'.xls') 

            # for i in range(len(images)):
            #   print('predict image: {}'.format(img_name[i]))

            # if model_type == 'agnet':
            #     np.save(os.path.join(model_score_dir, img_name[i].split('.')[0] + '.npy'), p_seg[i][1].cpu().numpy().astype(np.float32))
            #     cv2.imwrite(os.path.join(model_score_dir, img_name[i]), p_seg[i][1].cpu().numpy().astype(np.float32))
            # else:
            #     np.save(os.path.join(model_score_dir, img_name[i].split('.')[0] + '.npy'), p_seg[i][0].cpu().numpy().astype(np.float32))
            #     cv2.imwrite(os.path.join(model_score_dir, img_name[i].split('.')[0] + '.tif'), p_seg[i][0].cpu().numpy().astype(np.float32))




if __name__ == "__main__":
    opts = dict()
    opts['dataset_type'] = 'mito'

    opts["eval_batch_size"] = 32
    opts["gpu_list"] = "0,1,2,3"
    opts["train_dir"] = "./train_log/nucleus_train_unet_20210424_iouloss/checkpoints"
    opts["eval_data_dir"] = "./datasets/test_nucleus_new_selected.txt"
    # model_type = [unet, unetPlus, agnet, deeplab, penet]
    opts['model_type'] = 'unet'
    opts["load_epoch"] = 30

    eval_model(opts)
