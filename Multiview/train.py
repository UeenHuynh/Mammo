#coding=utf-8
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import shutil
import time
from utils.set_seeds import seed_everything
from utils.read_dataset import read_dataset
from utils.train_model import train
from config import seed, batch_size, root, model_path, init_lr, lr_decay_rate,\
    lr_milestones, weight_decay, end_epoch, dataset_path, input_size
from utils.auto_load_resume import auto_load_resume
import os
import argparse
from pytorch_metric_learning import losses, miners
from losses.focal import FocalLoss

from models.fusion import fusion_resnet18, fusion_resnet34, fusion_resnet50, fusion_resnet101, fusion_resnet152, \
                          fusion_resnext50_32x4d, fusion_resnext101_32x8d, fusion_resnext101_64x4d, \
                          fusion_wide_resnet50_2, fusion_wide_resnet101_2
from models.fusion import fusion_vgg11, fusion_vgg13, fusion_vgg16, fusion_vgg19

# from models.sg_prj_resnet import sg_prj_resnet50, sg_prj_resnet101, sg_prj_resnet152, sg_prj_resnext50_32x4d, sg_prj_resnext101_32x8d, sg_prj_resnext101_64x4d,\
#     sg_prj_wide_resnet50_2, sg_prj_wide_resnet101_2
# from models.mt_prj_resnet import mt_prj_resnet50, mt_prj_resnet101, mt_prj_resnet152, mt_prj_resnext50_32x4d, mt_prj_resnext101_32x8d, mt_prj_resnext101_64x4d,\
#     mt_prj_wide_resnet50_2, mt_prj_wide_resnet101_2


device = torch.device("cuda")
# device = torch.device("cpu")
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="chosen model")
ap.add_argument("-t", "--modeltype", required=True, help="chosen model type")
ap.add_argument("-agg", "--aggregation", required=True, help="chosen aggregation")
args = vars(ap.parse_args())

model_pool = {
    'resnet18': fusion_resnet18,
    'resnet34':fusion_resnet34,
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
pretrained_url_pool.update(dict.fromkeys(['resnet34'], 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'))
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
aggregation.update(dict.fromkeys(['w_average'], "w_avg"))


def main():
    # set all the necessary seeds
    seed_everything(seed)
    # Read the dataset
    # trainloader, valloader, testloader = read_dataset(input_size, batch_size, root, dataset_path)
    trainloader, testloader = read_dataset(input_size, batch_size, root, dataset_path)

    model = model_pool.get(args["model"])(pth_url=pretrained_url_pool.get(args["model"]), model_type = model_t.get(args["modeltype"]),
                                          aggregation = aggregation.get(args["aggregation"]),
                                          num_l = number_layers.get(args["model"]), pretrained=True)


    # define the CE loss function
    # criterion = nn.CrossEntropyLoss()
    # alpha = torch.tensor([1.0, 3.0, 300.0])
    alpha = torch.tensor([1.0, 4.0])
    alpha_CMMD = torch.tensor([2.5, 1.0])
    criterion = FocalLoss(alpha_CMMD.to(device))
    metric_loss = losses.TripletMarginLoss(0.2)
    miner = miners.BatchHardMiner()
    parameters = model.parameters()

    # define the optimizer
    optimizer = torch.optim.SGD(parameters, lr=init_lr, weight_decay=weight_decay)
    # define the learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate, verbose=True)

    # loading checkpoint
    save_path = os.path.join(model_path, f'{args["model"]}_{args["modeltype"]}_{args["aggregation"]}')
    if os.path.exists(save_path):
        start_epoch, best_f1 = auto_load_resume(model, optimizer, scheduler, save_path, status='train', device=device)
        assert start_epoch < end_epoch
    else:
        os.makedirs(save_path)
        best_f1 = 0.0
        start_epoch = 0

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    

    time_str = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy('./config.py', os.path.join(save_path, "{}config.py".format(time_str)))
     # Train the model
    train(model=model,
          device=device,
          trainloader=trainloader,
        #   valloader=valloader,
          testloader=testloader,
          metric_loss=metric_loss,
          miner=miner,
          criterion=criterion,
          optimizer=optimizer,
          scheduler=scheduler,
          save_path=save_path,
          start_epoch=start_epoch,
          end_epoch=end_epoch,
          best_f1 = best_f1)


if __name__ == '__main__':
    main()