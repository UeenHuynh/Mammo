import torch
import torch.nn as nn
import numpy as np 
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import imageio

from torch.optim.lr_scheduler import MultiStepLR
import time
from config import seed, batch_size, root, model_path, init_lr, lr_decay_rate,\
    lr_milestones, weight_decay, end_epoch, dataset_path, input_size
from utils.auto_load_resume import auto_load_resume
import os
from models.fusion import fusion_resnet18, fusion_resnet50, fusion_resnet101, fusion_resnet152, \
                          fusion_resnext50_32x4d, fusion_resnext101_32x8d, fusion_resnext101_64x4d, \
                          fusion_wide_resnet50_2, fusion_wide_resnet101_2
from models.fusion import fusion_vgg11, fusion_vgg13, fusion_vgg16, fusion_vgg19
import argparse
import matplotlib.pyplot as plt

# device = torch.device("cuda")
device = torch.device("cpu")

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="chosen model")
ap.add_argument("-t", "--modeltype", required=True, help="chosen model type")
ap.add_argument("-agg", "--aggregation", required=True, help="chosen aggregation")
args = vars(ap.parse_args())

model_pool = {
    'resnet18': fusion_resnet18,
    'resnet50': fusion_resnet50,
    'resnet101': fusion_resnet101,
    'resnet152': fusion_resnet152,
    'resnext50_32x4d': fusion_resnext50_32x4d,
    'resnext101_32x8d': fusion_resnext101_32x8d,
    'resnext101_64x4d': fusion_resnext101_64x4d,
    'wide_resnet50_2': fusion_wide_resnet50_2,
    'wide_resnet101_2': fusion_wide_resnet101_2,
    'vgg11': fusion_vgg11,
    'vgg13': fusion_vgg13,
    'vgg16': fusion_vgg16,
    'vgg19': fusion_vgg19,
}

number_layers = {
    'vgg11': [1,1,2,2,2],
    'vgg13': [2,2,2,2,2],
    'vgg16': [2,2,3,3,3],
    'vgg19': [2,2,4,4,4],
}


pretrained_url_pool = dict.fromkeys(['resnet50'], "https://download.pytorch.org/models/resnet50-11ad3fa6.pth")
pretrained_url_pool.update(dict.fromkeys(['resnet18'], "https://download.pytorch.org/models/resnet18-5c106cde.pth"))
pretrained_url_pool.update(dict.fromkeys(['resnet101'], "https://download.pytorch.org/models/resnet101-cd907fc2.pth"))
pretrained_url_pool.update(dict.fromkeys(['resnet152'], "https://download.pytorch.org/models/resnet152-f82ba261.pth"))
pretrained_url_pool.update(dict.fromkeys(['resnext50_32x4d'], "https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth"))
pretrained_url_pool.update(dict.fromkeys(['resnext101_32x8d'], "https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth"))
pretrained_url_pool.update(dict.fromkeys(['resnext101_64x4d'], "https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth"))
pretrained_url_pool.update(dict.fromkeys(['wide_resnet50_2'], "https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth"))
pretrained_url_pool.update(dict.fromkeys(['wide_resnet101_2'], "https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth"))
pretrained_url_pool.update(dict.fromkeys(['vgg11'], "https://download.pytorch.org/models/vgg11_bn-6002323d.pth"))
pretrained_url_pool.update(dict.fromkeys(['vgg13'], "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth"))
pretrained_url_pool.update(dict.fromkeys(['vgg16'], "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"))
pretrained_url_pool.update(dict.fromkeys(['vgg19'], "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth"))

model_t =  dict.fromkeys(['EF'], "EF")
model_t.update(dict.fromkeys(['PreF'], "PreF"))
model_t.update(dict.fromkeys(['MF'], "MF"))
model_t.update(dict.fromkeys(['LF'], "LF"))
model_t.update(dict.fromkeys(['PostF'], "PostF"))

aggregation =  dict.fromkeys(['average'], "avg")
aggregation.update(dict.fromkeys(['concat'], "cat"))


# class SingleImageNetwork(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.backbone = model

#     def __call__(self, x):
#         return self.backbone(x[0],x[1])

def main():

    # Read the dataset
    # trainloader, valloader, testloader = read_dataset(input_size, batch_size, root, dataset_path)
    # trainloader, testloader = read_dataset(input_size, batch_size, root, dataset_path)

    model = model_pool.get(args["model"])(pth_url=pretrained_url_pool.get(args["model"]), model_type = model_t.get(args["modeltype"]),
                                          aggregation = aggregation.get(args["aggregation"]),
                                          num_l = number_layers.get(args["model"]), pretrained=True)

    cc_img = imageio.imread(f'{dataset_path}/843abf53ca13b08410085e28fe4de489_15c8e59fdfafb3deefc80c5a9e8a42d0.png')
    mlo_img = imageio.imread(f'{dataset_path}/843abf53ca13b08410085e28fe4de489_83be060130997ca7b67b3979978a5d29.png')
    # cc_img = imageio.imread(f'{dataset_path}/e081a66ddcbc1f1856e1e6ed8f1a9bd9_86b8c44955993fd65239e9f9530cd7da.png')
    # mlo_img = imageio.imread(f'{dataset_path}/e081a66ddcbc1f1856e1e6ed8f1a9bd9_5fbfdd8f687474b4eb18cb6b3d840973.png')

    input_cc = torch.Tensor(cc_img)
    input_cc = torch.movedim(input_cc, 2, 0).unsqueeze(0)
    input_mlo= torch.Tensor(mlo_img)
    input_mlo = torch.movedim(input_mlo, 2, 0).unsqueeze(0)
    input_tensor = input_cc, input_mlo
    parameters = model.parameters()

    # define the optimizer
    optimizer = torch.optim.SGD(parameters, lr=init_lr, weight_decay=weight_decay)
    # define the learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate, verbose=True)

    # loading checkpoint
    save_path = os.path.join(model_path, f'{args["model"]}_{args["modeltype"]}_{args["aggregation"]}')
    if os.path.exists(save_path):
        start_epoch, best_f1= auto_load_resume(model, optimizer, scheduler, save_path, status='train', device=device)
        # assert start_epoch < end_epoch
    else:
        os.makedirs(save_path)
        best_f1 = 0.0
        start_epoch = 0

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model).to(device)
    # else:
    #     model = model.to(device)
    

    time_str = time.strftime("%Y%m%d-%H%M%S")
    
    # print(model.early_layers_l)
    # print('Early Layer CC: ',model.early_layers_l[0].weight)
    # print('Early Layer MLO: ',model.early_layers_r[0].weight)
    # print('block 64 shape: ',model.early_layers_r[0].weight.shape)
    # a = sum(sum(model.early_layers_l[0].weight == model.early_layers_r[0].weight))
    # print('sum shape: ',a.shape)
    # print('sum: ',a)
    # print(model.early_layers_l[-2][-2])
    outputs_cc = []
    name_cc = []
    outputs_mlo = []
    name_mlo = []
    # for i in range(1,3):
    #     for layer in model.early_layers_l[-i]:
    #         f_cc = layer(input_cc)
    #         outputs_cc.append(f_cc)
    #         name_cc.append(str(layer))
    #     for layer in model.early_layers_r[-i]:
    #         f_mlo = layer(input_mlo)
    #         outputs_mlo.append(f_mlo)
    #         name_mlo.append(str(layer))

    # f_cc =  model.early_layers_l(input_cc)

    # name_cc.append(str(model.early_layers_l[0]))

    # f_mlo = model.early_layers_r(input_mlo)
    # outputs_mlo.append(f_mlo)
    # name_mlo.append(str(model.early_layers_r[0]))

    # f_cc = (f_cc + f_mlo)/2
    # f_cc = model.conv_avg(f_cc)
    # outputs_cc.append(f_cc)
    
    # processed_cc = []
    # for feature_map in outputs_cc:
    #     feature_map = feature_map.squeeze(0)
    #     gray_scale = torch.sum(feature_map,0)
    #     gray_scale = gray_scale / feature_map.shape[0]
    #     processed_cc.append(gray_scale.data.cpu().numpy())

    # processed_mlo = []
    # for feature_map in outputs_mlo:
    #     feature_map = feature_map.squeeze(0)
    #     gray_scale = torch.sum(feature_map,0)
    #     gray_scale = gray_scale / feature_map.shape[0]
    #     processed_mlo.append(gray_scale.data.cpu().numpy())

    # fig = plt.figure(figsize=(30, 50))
    # for i in range(len(processed_cc)):
    #     a = fig.add_subplot(5, 4, i+1)
    #     imgplot = plt.imshow(processed_cc[i])
    #     a.axis("off")
    #     a.set_title(name_cc[i].split('(')[0], fontsize=30)
    # plt.savefig(str('feature_maps_cc_block2_2.jpg'), bbox_inches='tight')

    # fig = plt.figure(figsize=(30, 50))
    # for i in range(len(processed_mlo)):
    #     a = fig.add_subplot(5, 4, i+1)
    #     imgplot = plt.imshow(processed_mlo[i])
    #     a.axis("off")
    #     a.set_title(name_mlo[i].split('(')[0], fontsize=30)
    # plt.savefig(str('feature_maps_mlo_block2_2.jpg'), bbox_inches='tight')

    # target_layers = [model.last_layers[1][-1]]
    target_layers = [model.early_layers_mlo[-1][-2]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    targets = [ClassifierOutputTarget(1)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(np.float32(cc_img)/255, grayscale_cam, use_rgb=True)
    imageio.imwrite(f'grad_cam_CC_MLO4_{args["model"]}_{args["modeltype"]}_{args["aggregation"]}.png', visualization, format=None)
    
    target_layers = [model.early_layers_cc[-1][-2]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(np.float32(cc_img)/255, grayscale_cam, use_rgb=True)
    imageio.imwrite(f'grad_cam_CC_CC4_{args["model"]}_{args["modeltype"]}_{args["aggregation"]}.png', visualization, format=None)

if __name__ == '__main__':
    main()