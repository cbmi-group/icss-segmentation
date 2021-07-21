#from models.DeepLab.modeling.backbone import resnet, xception, drn, mobilenet
from models.DeepLab.modeling.backbone import resnet_modified, xception, drn, mobilenet
def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet101':
        return resnet_modified.ResNet101(output_stride, BatchNorm)
    elif backbone == 'resnet50':
        return resnet_modified.ResNet50(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
