import torch
import torch.nn as nn
import numpy as np 
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, resnext101_64x4d, wide_resnet50_2, wide_resnet101_2
from models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from collections import OrderedDict
import torch.nn.functional as F
from config import dataset_path
import imageio

# cc_img = imageio.imread(f'{dataset_path}/843abf53ca13b08410085e28fe4de489_15c8e59fdfafb3deefc80c5a9e8a42d0.png')
# mlo_img = imageio.imread(f'{dataset_path}/843abf53ca13b08410085e28fe4de489_83be060130997ca7b67b3979978a5d29.png')

# input_tensor = torch.Tensor(mlo_img)
# input_tensor = torch.movedim(input_tensor, 2, 0).unsqueeze(0)
class WeightedFusionNet(nn.Module):
    def __init__(self, args):
        super(WeightedFusionNet, self).__init__()

        self.args = args

        self.net1 = nn.Sequential(OrderedDict([
            ('fc', nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, bias=False)),
            ('relu', nn.LeakyReLU()),
            ('fc_weight', nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, bias=False))
        ]))

        self.net2 = nn.Sequential(OrderedDict([
            ('fc', nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, bias=False)),
            ('relu', nn.LeakyReLU()),
            ('fc_weight', nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, bias=False))
        ]))
    
    def forward(self, feat1, feat2):
        w1 = self.net1(feat1.transpose(1,2).unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        w2 = self.net2(feat2.transpose(1,2).unsqueeze(-1)).transpose(1, 2).squeeze(-1)
        w = F.softmax(torch.cat((w1, w2), dim=-1), dim=-1)
        w1, w2 = w[:,:,0].unsqueeze(-1).repeat(1, 1, feat1.shape[-1]), w[:,:,1].unsqueeze(-1).repeat(1, 1, feat1.shape[-1])

        if self.args.aggregation == 'net_c':
            feat1 = feat1 * w1
            feat2 = feat2 * w2
            feat = torch.cat((feat1, feat2), dim=-1)
        else:
            feat = feat1 * w1 + feat2 * w2

        return feat, w1, w2
    
class Fusion_ResNet(nn.Module):
    def __init__(self, backbone, model_type, aggregation, num_classes=2, dropout = 0.2):
        super().__init__()
        self.aggregation = aggregation
        self.backbone = backbone
        self.model_type = model_type
        self.fc_expand = 1

        early_layers_cc = []
        early_layers_mlo = []
        last_layers = []
        #Weighted Fusion Network

        self.net1 = nn.Sequential(OrderedDict([
            ('fc', nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False)),
            ('relu', nn.LeakyReLU()),
            ('fc_weight', nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, bias=False))
        ]))

        self.net2 = nn.Sequential(OrderedDict([
            ('fc', nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False)),
            ('relu', nn.LeakyReLU()),
            ('fc_weight', nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, bias=False))
        ]))

        child_counter = 0
        if model_type == "PreF":
            for child in self.backbone.children():
                if child_counter < 0:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    last_layers.append(child)
                child_counter += 1
            
            out_n = 3
            out_fc = last_layers[-1].out_features
            self.out_dim =  out_n

        if model_type == "EF":
            for child in self.backbone.children():
                if child_counter <= 3:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    last_layers.append(child)
                child_counter += 1
            
            out_n = early_layers_cc[0].out_channels
            out_fc = last_layers[-1].out_features
            self.out_dim =  out_n

        if model_type == "MF":
            for child in self.backbone.children():
                if child_counter <= 5:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    last_layers.append(child)
                child_counter += 1

            out_n =  early_layers_cc[-1][0].conv1.out_channels
            out_fc = last_layers[-1].out_features
            self.out_dim =  out_n * self.backbone.block.expansion

        if model_type == "LF":
            for child in self.backbone.children():
                if child_counter <= 7:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    last_layers.append(child)
                child_counter += 1
            out_n =  early_layers_cc[-1][0].conv1.out_channels
            out_fc = last_layers[-1].out_features
            self.out_dim =  out_n * self.backbone.block.expansion

        if model_type == "PostF":
            for child in self.backbone.children():
                if child_counter <= 11:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    last_layers.append(child)
                child_counter += 1
            out_n =  3
            out_fc = early_layers_cc[-1].out_features
            if self.aggregation == 'cat':
                self.fc_expand = 2
            self.out_dim =  out_n * self.backbone.block.expansion
        
        self.conv_avg = nn.Conv2d(self.out_dim , self.out_dim, kernel_size=1, bias=False)
        self.conv_ccat = nn.Conv2d(self.out_dim*2 , self.out_dim , kernel_size=1, bias=False)

        self.conv1 = nn.Conv2d(self.out_dim , self.out_dim, kernel_size=1, bias=False) 
        self.bn = nn.BatchNorm2d(self.out_dim)

        self.early_layers_cc = nn.Sequential(*early_layers_cc)
        self.early_layers_mlo = nn.Sequential(*early_layers_mlo)
        self.last_layers = nn.Sequential(*last_layers)

        self.fc = nn.Linear(out_fc * self.fc_expand, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()

    def _forward_implement(self, cc, mlo):
        
        f_cc = self.early_layers_cc(cc)
        f_mlo = self.early_layers_mlo(mlo)
        # print(f_cc.shape)
        if self.aggregation == 'w_avg':
            w1 = self.net1(f_cc.transpose(0,1).unsqueeze(-1))
            w2 = self.net2(f_mlo.transpose(0,1).unsqueeze(-1))
            # print(w1.shape) e dung di, a hoc tieng anh xiu. Vl :v ok a, e dang de train. Vay e dien so cho bang thuc nghiem tiep =)) xong r e qua lam Inbreast cho xong Ok, a` em in cai weight ra di. Oke a`
            w = F.softmax(torch.cat((w1, w2), dim=-1), dim=-1)
            # print(w.shape)
            w1, w2 = w[:,:,0].unsqueeze(-1).repeat(1, 1, f_cc.shape[-1]), w[:,:,1].unsqueeze(-1).repeat(1, 1, f_cc.shape[-1])
            # print(w1.shape)
            w1, w2 = w1.squeeze(0), w2.squeeze(0)
            feat1 = f_cc * w1
            feat2 = f_mlo * w2
            feat = feat1 * w1 + feat2 * w2
            # print("w1: ", w1.mean())
            # print("\nw2: ", w2.mean())
            x = self.last_layers(feat)
            logits = self.fc(x)
        if self.aggregation == 'avg':
            x = (f_cc + f_mlo)/2
            if self.model_type in ['PreF', 'EF', 'MF', 'LF']:
                x = self.conv_avg(x)
                x = self.bn(x)
                x = self.relu(x)
                x = x + f_mlo
            x = self.last_layers(x)
            logits = self.fc(x)
            # x = self.relu(x)
            # x = self.dropout(x)
            # logits = self.softmax(x) 

        if self.aggregation == 'cat':
            x = torch.cat((f_cc,f_mlo),1)
            if self.model_type in ['PreF', 'EF', 'MF', 'LF']:
                x = self.conv_ccat(x)
                x = self.bn(x)
                x = self.relu(x)
                x = x + f_mlo
            x = self.last_layers(x)
            logits = self.fc(x)
            # x = self.relu(x)
            # x = self.dropout(x)
            # logits = self.softmax(x)

        return logits

    def forward(self, cc, mlo):
        logits = self._forward_implement(cc, mlo)
        return logits

class Fusion_VGG(nn.Module):
    def __init__(self, backbone, model_type, aggregation, num_l, num_classes=2, dropout = 0.2):
        super().__init__()
        self.aggregation = aggregation
        self.model_type = model_type
        self.fc_expand = 1

        early_layers_cc = []
        early_layers_mlo = []
        mid_layers = []
        last_layers = []

        self.layers = num_l
        child_counter = 0

        if model_type == "PreF":
            for child in backbone.features:
                if child_counter < 0:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    mid_layers.append(child)
                child_counter += 1

            self.out_dim = 3

        if model_type == "EF":
            v = 3*sum(self.layers[:1]) + len(self.layers[:1])
            for child in backbone.features:
                if child_counter < v:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    mid_layers.append(child)
                child_counter += 1

            self.out_dim =  early_layers_cc[0].out_channels

        if model_type == "MF":
            v = 3*sum(self.layers[:3]) + len(self.layers[:3])
            for child in backbone.features:
                if child_counter < v:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    mid_layers.append(child)
                child_counter += 1

            self.out_dim =  early_layers_cc[-4].out_channels

        if model_type == "LF" or model_type == "PostF" :
            v = 3*sum(self.layers[:5]) + len(self.layers[:5])
            for child in backbone.features:
                if child_counter < v:
                    early_layers_cc.append(child)
                    early_layers_mlo.append(child)
                else:
                    mid_layers.append(child)
                child_counter += 1

            self.out_dim =  early_layers_cc[-4].out_channels

        
        self.conv_avg = nn.Conv2d(self.out_dim , self.out_dim, kernel_size=1, bias=False)
        self.conv_ccat = nn.Conv2d(self.out_dim*2 , self.out_dim , kernel_size=1, bias=False)

        if model_type == "PostF":
            early_layers_cc.append(backbone.avgpool)
            early_layers_cc.append(backbone.flatten)
            early_layers_cc.append(backbone.classifier)
            early_layers_mlo.append(backbone.avgpool)
            early_layers_mlo.append(backbone.flatten)
            early_layers_mlo.append(backbone.classifier)
            self.early_layers_l = nn.Sequential(*early_layers_cc)
            self.early_layers_r = nn.Sequential(*early_layers_mlo)
            
            if self.aggregation == 'cat':
                self.fc_expand = 2
            out_fc = self.early_layers[-1][-1].out_features
        else:
            self.early_layers_l = nn.Sequential(*early_layers_cc)
            self.early_layers_r = nn.Sequential(*early_layers_mlo)
            self.mid_layers = nn.Sequential(*mid_layers)

            last_layers.append(self.mid_layers)
            last_layers.append(backbone.avgpool)
            last_layers.append(backbone.flatten)
            last_layers.append(backbone.classifier)
            self.last_layers = nn.Sequential(*last_layers)

            out_fc = self.last_layers[-1][-1].out_features

        self.fc = nn.Linear(out_fc * self.fc_expand, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _forward_implement(self, cc, mlo):
        f_cc = self.early_layers_cc(cc)
        f_mlo = self.early_layers_mlo(mlo)
        if self.aggregation == 'avg':
            x = (f_cc + f_mlo)/2
            if self.model_type in ['PreF', 'EF', 'MF', 'LF']:
                x = self.conv_avg(x)
                x = self.bn(x)
                x = self.relu(x)
                x = x + f_mlo
            x = self.last_layers(x)
            logits = self.fc(x)
            # logits = self.softmax(x) 

        if self.aggregation == 'cat':
            x = torch.cat((f_cc,f_mlo),1)
            if self.model_type in ['PreF', 'EF', 'MF', 'LF']:
                x = self.conv_ccat(x)
                x = self.bn(x)
                x = self.relu(x)
                x = x + f_mlo
            x = self.last_layers(x)
            logits = self.fc(x)
            # logits = self.softmax(x)
        
        return logits

    def forward(self, cc, mlo):
        logits = self._forward_implement(cc, mlo)
        return logits


#Fusion for VGG Family  

def fusion_vgg11(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_VGG(vgg11_bn(pth_url, pretrained), model_type, aggregation, num_l, **kwargs)

def fusion_vgg13(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_VGG(vgg13_bn(pth_url, pretrained), model_type, aggregation, num_l, **kwargs)

def fusion_vgg16(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_VGG(vgg16_bn(pth_url, pretrained), model_type, aggregation, num_l, **kwargs)

def fusion_vgg19(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_VGG(vgg19_bn(pth_url, pretrained), model_type, aggregation, num_l, **kwargs)

#Fusion for ResNet Family

def fusion_resnet18(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnet18(pth_url, pretrained), model_type, aggregation, **kwargs)

def fusion_resnet34(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnet34(pth_url, pretrained), model_type, aggregation, **kwargs)


def fusion_resnet50(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnet50(pth_url, pretrained), model_type, aggregation, **kwargs)


def fusion_resnet101(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnet101(pth_url, pretrained), model_type, aggregation, **kwargs)


def fusion_resnet152(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnet152(pth_url, pretrained), model_type, aggregation, **kwargs)


def fusion_resnext50_32x4d(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnext50_32x4d(pth_url, pretrained), model_type, aggregation, **kwargs)


def fusion_resnext101_32x8d(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnext101_32x8d(pth_url, pretrained), model_type, aggregation, **kwargs)


def fusion_resnext101_64x4d(pth_url,model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(resnext101_64x4d(pth_url, pretrained), model_type, aggregation, **kwargs)


def fusion_wide_resnet50_2(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(wide_resnet50_2(pth_url, pretrained), model_type, aggregation, **kwargs)


def fusion_wide_resnet101_2(pth_url, model_type, aggregation, num_l, pretrained=False, **kwargs):
    return Fusion_ResNet(wide_resnet101_2(pth_url, pretrained), model_type, aggregation, **kwargs)








