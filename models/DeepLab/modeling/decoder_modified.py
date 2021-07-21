
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DeepLab.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
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
class Decoder_plus(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder_plus, self).__init__()
        if backbone == 'resnet101' or backbone == 'drn' or backbone == 'resnet50':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError
        self.conv0 = nn.Conv2d(64, 64, 1, bias=False)
        self.bn0 = BatchNorm(64)
        self.conv1 = nn.Conv2d(256, 128, 1, bias=False)
        self.bn1 = BatchNorm(128)
        self.conv2 = nn.Conv2d(512, 256, 1, bias=False)
        self.bn2 = BatchNorm(256)
        self.conv3 = nn.Conv2d(1024, 256, 1, bias=False)
        self.bn3 = BatchNorm(256)
        self.conv4 = nn.Conv2d(2048, 128, 1, bias=False)
        self.bn4 = BatchNorm(128)
        self.conv_x3 = DoubleConv(128, 64)
        self.conv_x2 = DoubleConv(256, 128)
        self.conv_x1 = DoubleConv(256, 256)

        self.relu = nn.ReLU()
        # Sequential
        self.last_conv = nn.Sequential(nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       # BatchNorm
                                       BatchNorm(256),
                                       # ReLU()
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat0, low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4):

        # low_level_feat4 = self.conv4(low_level_feat4)
        # low_level_feat4 = self.bn1(low_level_feat4)
        # low_level_feat4 = self.relu(low_level_feat4)
        # x = torch.cat((x, low_level_feat4), dim=1)
        # x=self.conv_x1(x)
        # print('low_level_feat0 size',low_level_feat0.shape)
        # print('low_level_feat1 size',low_level_feat1.shape)
        # print('low_level_feat2 size',low_level_feat2.shape)
        # print('low_level_feat3 size',low_level_feat3.shape)
        # print('x size',x.shape)

        # x = F.interpolate(x, size=low_level_feat3.size()[2:], mode='bilinear', align_corners=True)
        low_level_feat3 = self.conv3(low_level_feat3)
        low_level_feat3 = self.bn3(low_level_feat3)
        low_level_feat3 = self.relu(low_level_feat3)
        x = x+low_level_feat3

        x = self.conv_x1(x)

        x = F.interpolate(x, size=low_level_feat2.size()[2:], mode='bilinear', align_corners=True)
        low_level_feat2 = self.conv2(low_level_feat2)
        low_level_feat2 = self.bn2(low_level_feat2)
        low_level_feat2 = self.relu(low_level_feat2)
        x =x+ low_level_feat2
        x = self.conv_x2(x)

        x = F.interpolate(x, size=low_level_feat1.size()[2:], mode='bilinear', align_corners=True)
        low_level_feat1 = self.conv1(low_level_feat1)
        low_level_feat1 = self.bn1(low_level_feat1)
        low_level_feat1 = self.relu(low_level_feat1)
        x = x+low_level_feat1
        x = self.conv_x3(x)

        x = F.interpolate(x, size=low_level_feat0.size()[2:], mode='bilinear', align_corners=True)
        low_level_feat0 = self.conv0(low_level_feat0)
        low_level_feat0 = self.bn0(low_level_feat0)
        low_level_feat0 = self.relu(low_level_feat0)
        x = x+ low_level_feat0

        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet101' or backbone == 'drn' or backbone == 'resnet50':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError
        self.conv0 = nn.Conv2d(64, 64, 1, bias=False)
        self.bn0 = BatchNorm(64)
        self.conv1 = nn.Conv2d(256, 128, 1, bias=False)
        self.bn1 = BatchNorm(128)
        self.conv2 = nn.Conv2d(512, 256, 1, bias=False)
        self.bn2 = BatchNorm(256)
        self.conv3 = nn.Conv2d(1024, 256, 1, bias=False)
        self.bn3 = BatchNorm(256)
        self.conv4 = nn.Conv2d(2048, 128, 1, bias=False)
        self.bn4 = BatchNorm(128)
        self.conv_x3 = DoubleConv(256, 64)
        self.conv_x2 = DoubleConv(512, 128)
        self.conv_x1 = DoubleConv(512, 256)

        self.relu = nn.ReLU()
        # Sequential
        self.last_conv = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       # BatchNorm
                                       BatchNorm(256),
                                       # ReLU()
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat0, low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4):

        # low_level_feat4 = self.conv4(low_level_feat4)
        # low_level_feat4 = self.bn1(low_level_feat4)
        # low_level_feat4 = self.relu(low_level_feat4)
        # x = torch.cat((x, low_level_feat4), dim=1)
        # x=self.conv_x1(x)
        # print('low_level_feat0 size',low_level_feat0.shape)
        # print('low_level_feat1 size',low_level_feat1.shape)
        # print('low_level_feat2 size',low_level_feat2.shape)
        # print('low_level_feat3 size',low_level_feat3.shape)
        # print('x size',x.shape)

        # x = F.interpolate(x, size=low_level_feat3.size()[2:], mode='bilinear', align_corners=True)
        low_level_feat3 = self.conv3(low_level_feat3)
        low_level_feat3 = self.bn3(low_level_feat3)
        low_level_feat3 = self.relu(low_level_feat3)
        x = torch.cat((x, low_level_feat3), dim=1)

        x = self.conv_x1(x)

        x = F.interpolate(x, size=low_level_feat2.size()[2:], mode='bilinear', align_corners=True)
        low_level_feat2 = self.conv2(low_level_feat2)
        low_level_feat2 = self.bn2(low_level_feat2)
        low_level_feat2 = self.relu(low_level_feat2)
        x = torch.cat((x, low_level_feat2), dim=1)
        x = self.conv_x2(x)

        x = F.interpolate(x, size=low_level_feat1.size()[2:], mode='bilinear', align_corners=True)
        low_level_feat1 = self.conv1(low_level_feat1)
        low_level_feat1 = self.bn1(low_level_feat1)
        low_level_feat1 = self.relu(low_level_feat1)
        x = torch.cat((x, low_level_feat1), dim=1)
        x = self.conv_x3(x)

        x = F.interpolate(x, size=low_level_feat0.size()[2:], mode='bilinear', align_corners=True)
        low_level_feat0 = self.conv0(low_level_feat0)
        low_level_feat0 = self.bn0(low_level_feat0)
        low_level_feat0 = self.relu(low_level_feat0)
        x = torch.cat((x, low_level_feat0), dim=1)

        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder_2(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder_2, self).__init__()
        if backbone == 'resnet101' or backbone == 'drn' or backbone == 'resnet50':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError
        self.conv0 = nn.Conv2d(64, 64, 1, bias=False)
        self.bn0 = BatchNorm(64)
        self.conv1 = nn.Conv2d(256, 128, 1, bias=False)
        self.bn1 = BatchNorm(128)
        self.conv2 = nn.Conv2d(512, 256, 1, bias=False)
        self.bn2 = BatchNorm(256)
        self.conv3 = nn.Conv2d(1024, 256, 1, bias=False)
        self.bn3 = BatchNorm(256)
        self.conv4 = nn.Conv2d(2048, 128, 1, bias=False)
        self.bn4 = BatchNorm(128)
        self.conv_x3 = DoubleConv(256, 64)
        self.conv_x2 = DoubleConv(512 ,128)
        self.conv_x1 = DoubleConv(512, 256)
        
        self.relu = nn.ReLU()
        # Sequential 
        self.last_conv = nn.Sequential(nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       # BatchNorm 
                                       BatchNorm(256),
                                       # ReLU() 
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat0, low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4):
    
        #low_level_feat4 = self.conv4(low_level_feat4)
        #low_level_feat4 = self.bn1(low_level_feat4)
        #low_level_feat4 = self.relu(low_level_feat4)
        #x = torch.cat((x, low_level_feat4), dim=1)
        #x=self.conv_x1(x)
        #print('low_level_feat0 size',low_level_feat0.shape)
        #print('low_level_feat1 size',low_level_feat1.shape)
        #print('low_level_feat2 size',low_level_feat2.shape)
        #print('low_level_feat3 size',low_level_feat3.shape)
        #print('x size',x.shape)

        #x = F.interpolate(x, size=low_level_feat3.size()[2:], mode='bilinear', align_corners=True)
        low_level_feat3 = self.conv3(low_level_feat3)
        low_level_feat3 = self.bn3(low_level_feat3)
        low_level_feat3 = self.relu(low_level_feat3)
        x = torch.cat((x, low_level_feat3), dim=1)
        
        x=self.conv_x1(x)

        x = F.interpolate(x, size=low_level_feat2.size()[2:], mode='bilinear', align_corners=True)
        low_level_feat2 = self.conv2(low_level_feat2)
        low_level_feat2 = self.bn2(low_level_feat2)
        low_level_feat2 = self.relu(low_level_feat2)
        x = torch.cat((x, low_level_feat2), dim=1)
        x=self.conv_x2(x)

        x = F.interpolate(x, size=low_level_feat1.size()[2:], mode='bilinear', align_corners=True)
        low_level_feat1 = self.conv1(low_level_feat1)
        low_level_feat1 = self.bn1(low_level_feat1)
        low_level_feat1 = self.relu(low_level_feat1)
        x = torch.cat((x, low_level_feat1), dim=1)
        x=self.conv_x3(x)
      

        #x = F.interpolate(x, size=low_level_feat0.size()[2:], mode='bilinear', align_corners=True)
        #low_level_feat0 = self.conv0(low_level_feat0)
        #low_level_feat0 = self.bn0(low_level_feat0)
        #low_level_feat0 = self.relu(low_level_feat0)
        #x = torch.cat((x, low_level_feat0), dim=1)
        
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)
def build_decoder_2(num_classes, backbone, BatchNorm):
    return Decoder_2(num_classes, backbone, BatchNorm)
def build_decoder_plus(num_classes, backbone, BatchNorm):
    return Decoder_plus(num_classes, backbone, BatchNorm)