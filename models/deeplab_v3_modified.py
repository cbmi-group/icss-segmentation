#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DeepLab.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.DeepLab.modeling.aspp import build_aspp
from models.DeepLab.modeling.decoder_modified import build_decoder,build_decoder_2,build_decoder_plus
from models.DeepLab.modeling.backbone import build_backbone


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DeepLab(nn.Module):
    def __init__(self, backbone, output_stride=16, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.conv_x1 = DoubleConv(320, 256)
        self.conv0 = nn.Conv2d(64, 48, 1, bias=False)
        self.bn0 = BatchNorm(48)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        # x, low_level_feat = self.backbone(input)
        
        x, low_level_feat0, low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4 = self.backbone(input)
        
       
        
        x = self.aspp(x)
        #print('x size:',x.shape)
        x = self.decoder(x, low_level_feat0, low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        output = torch.sigmoid(x)

     
        return output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

class DeepLab_plus(nn.Module):
    def __init__(self, backbone, output_stride=16, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab_plus, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder_plus(num_classes, backbone, BatchNorm)
        self.conv_x1 = DoubleConv(320, 256)
        self.conv0 = nn.Conv2d(64, 48, 1, bias=False)
        self.bn0 = BatchNorm(48)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        # x, low_level_feat = self.backbone(input)
        
        x, low_level_feat0, low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4 = self.backbone(input)
        
       
        
        x = self.aspp(x)
        #print('x size:',x.shape)
        x = self.decoder(x, low_level_feat0, low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        output = torch.sigmoid(x)

     
        return output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

class DeepLab_2(nn.Module):
    def __init__(self, backbone, output_stride=16, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab_2, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder_2(num_classes, backbone, BatchNorm)
        self.conv_x1 = DoubleConv(320, 256)
        self.conv0 = nn.Conv2d(64, 48, 1, bias=False)
        self.bn0 = BatchNorm(48)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        # x, low_level_feat = self.backbone(input)

        x, low_level_feat0, low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4 = self.backbone(input)

        x = self.aspp(x)
        # print('x size:',x.shape)
        x = self.decoder(x, low_level_feat0, low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        output = torch.sigmoid(x)

        return output

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


if __name__ == "__main__":
    model = DeepLab(backbone='resnet50', output_stride=16)
    model.eval()
    torch.manual_seed(1)
    input = torch.rand(1, 1, 128, 128)
    print(input)
    output = model(input)
    print(output)
    print(output.size())