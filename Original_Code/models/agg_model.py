import torch
import torch.nn as nn
# import timm
from models.resnet import resnet18, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, resnext101_64x4d, wide_resnet50_2, wide_resnet101_2


class FusionNet(nn.Module):
    def __init__(self, backbone, aggregation, num_classes=3, dropout = 0.2):
        super().__init__()
        self.aggregation = aggregation
        self.backbone = backbone
        child_counter = 0
        for child in self.backbone.children():
          print(" child", child_counter, "is:")
          print(child)
          child_counter += 1
        self.fc_avg = nn.Linear(512, 128)
        self.fc_cat = nn.Linear(1024, 128)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)
        self.have_prj = False

    def forward(self, lcc, lmlo):
        # if self.backbone == 'maxvit':
        #     self.backbone = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

        _, f_lcc = self.backbone(lcc)
        _, f_mlo = self.backbone(lmlo)
        if self.aggregation == 'avg':
            x = (f_lcc + f_mlo)/2
            x = self.fc_avg(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc(x)
            logits = self.softmax(x) 
        if self.aggregation == 'cat':
            x = torch.cat((f_lcc,f_mlo),1)
            x = self.fc_cat(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc(x)
            logits = self.softmax(x)
        
        return None, logits

# def agg_maxvit(pth_url, aggregation, pretrained=False, **kwargs):
#     model = torch.hub.load(pth_url, 'deit_base_patch16_224', pretrained=True)
#     print(model)
#     # return AggNet(pth_url, aggregation, **kwargs)
#     return model


def fusion_resnet18(pth_url, aggregation, pretrained=False, **kwargs):
    return FusionNet(resnet18(pth_url, pretrained), aggregation, **kwargs)


def fusion_resnet50(pth_url, aggregation, pretrained=False, **kwargs):
    return FusionNet(resnet50(pth_url, pretrained), aggregation, **kwargs)


def fusion_resnet101(pth_url, aggregation, pretrained=False, **kwargs):
    return FusionNet(resnet101(pth_url, pretrained), aggregation, **kwargs)


def fusion_resnet152(pth_url, aggregation, pretrained=False, **kwargs):
    return FusionNet(resnet152(pth_url, pretrained), aggregation, **kwargs)


def fusion_resnext50_32x4d(pth_url, aggregation, pretrained=False, **kwargs):
    return FusionNet(resnext50_32x4d(pth_url, pretrained), aggregation, **kwargs)


def fusion_resnext101_32x8d(pth_url, aggregation, pretrained=False, **kwargs):
    return FusionNet(resnext101_32x8d(pth_url, pretrained), aggregation, **kwargs)


def fusion_resnext101_64x4d(pth_url, aggregation, pretrained=False, **kwargs):
    return FusionNet(resnext101_64x4d(pth_url, pretrained), aggregation, **kwargs)


def fusion_wide_resnet50_2(pth_url, aggregation, pretrained=False, **kwargs):
    return FusionNet(agg_wide_resnet50_2(pth_url, pretrained), aggregation, **kwargs)


def fusion_wide_resnet101_2(pth_url, aggregation, pretrained=False, **kwargs):
    return FusionNet(wide_resnet101_2(pth_url, pretrained), aggregation, **kwargs)








