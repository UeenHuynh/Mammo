import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from imblearn.metrics import geometric_mean_score
from utils.read_dataset import read_dataset
from collections import OrderedDict
from tqdm import tqdm
from models.resnet import resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, resnext101_64x4d, wide_resnet50_2, wide_resnet101_2
from models.fusion import fusion_resnet18, fusion_resnet50, fusion_resnet101, fusion_resnet152, \
                          fusion_resnext50_32x4d, fusion_resnext101_32x8d, fusion_resnext101_64x4d, \
                          fusion_wide_resnet50_2, fusion_wide_resnet101_2
from models.fusion import fusion_vgg11, fusion_vgg13, fusion_vgg16, fusion_vgg19
# from models.sg_prj_resnet import sg_prj_resnet50, sg_prj_resnet101, sg_prj_resnet152, sg_prj_resnext50_32x4d, sg_prj_resnext101_32x8d, sg_prj_resnext101_64x4d,\
#     sg_prj_wide_resnet50_2, sg_prj_wide_resnet101_2
# from models.mt_prj_resnet import mt_prj_resnet50, mt_prj_resnet101, mt_prj_resnet152, mt_prj_resnext50_32x4d, mt_prj_resnext101_32x8d, mt_prj_resnext101_64x4d,\
#     mt_prj_wide_resnet50_2, mt_prj_wide_resnet101_2
from config import input_size, batch_size, root, dataset_path



# device = torch.device("cuda")
device = torch.device("cpu")

# change

pth_path = r"./checkpoint/resnet50/best_model.pth"
_, testloader = read_dataset(input_size, batch_size, root, dataset_path)

# change
model = fusion_resnet50(pth_url=pth_path, model_type = 'PF' , aggregation = 'cat', num_l = [1], pretrained=False)

checkpoint = torch.load(pth_path, map_location='cpu')
new_state_dict = OrderedDict()
for k, v in checkpoint['model_state_dict'].items():
    name = k[7:]
    new_state_dict[name] = v
model.load_state_dict(checkpoint['model_state_dict'])
if __name__ == '__main__':
    print("Model loaded!")
    model = model.to(device)

    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(tqdm(testloader)):
            img_lcc, img_lmlo, labels = data
            img_lcc = img_lcc.to(device) 
            img_lmlo = img_lmlo.to(device)
            labels = (labels-1).to(device)
            _, logits = model(img_lcc, img_lmlo)

            preds = logits.max(1, keepdim=True)[1]
            print(preds.shape)
            all_preds.append(preds)
            print(all_preds)
            all_labels.append(labels)
    print(np.shape(all_preds))
    all_preds = torch.cat(all_preds, axis=0)
    print(all_preds.shape)
    all_labels = torch.cat(all_labels, axis=0)
    all_preds, all_labels = all_preds.cpu(), all_labels.cpu()
    
    # constant for classes
    classes = ('Normal', 'Benign','Recall')
    # classes = ('Density A','Density B','Density C','Density D')
    # Build confusion matrix
    cf_matrix = confusion_matrix(all_labels, all_preds)

    df_cm = pd.DataFrame(cf_matrix , index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
    matrix = sn.heatmap(cmn, annot=True, fmt='.2f', xticklabels=[i for i in classes], yticklabels=[i for i in classes])
    plt.title('Confusion Matrix') 
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.plot()
    plt.savefig(r"./checkpoint/resnet50/cfmatrix.png")

    print(accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro'))