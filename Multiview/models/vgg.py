import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes = 128, init_weights = True, dropout = 0.5):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm = False) -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    
    return nn.Sequential(*layers)

cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
        
def _vgg(pth_url, cfg, batch_norm, pretrained = False, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        pretrained_dict = torch.hub.load_state_dict_from_url(pth_url)
        # print(pretrained_dict.keys())
        del pretrained_dict['classifier.6.weight']
        del pretrained_dict['classifier.6.bias']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def vgg11_bn(pth_url, pretrained):
    return _vgg(pth_url, cfg = "A", batch_norm = True,  pretrained = pretrained)

def vgg13_bn(pth_url, pretrained):
    return _vgg(pth_url, cfg = "B", batch_norm = True,  pretrained = pretrained)

def vgg16_bn(pth_url, pretrained):
    return _vgg(pth_url, cfg = "D", batch_norm = True,  pretrained = pretrained)

def vgg19_bn(pth_url, pretrained):
    return _vgg(pth_url, cfg = "E", batch_norm = True,  pretrained = pretrained)



