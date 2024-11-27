import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np 
from collections import Counter

def eval(model, device, loader, metric_loss, miner, criterion, split):
    model.eval()
    print('Evaluating model on ' + split + ' data')

    ce_loss_sum = 0
    metric_loss_sum = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            img_cc, img_mlo, label = data
            img_cc, img_mlo = img_cc.to(device), img_mlo.to(device)
            label = (label-1).to(device)

            logits = model(img_cc, img_mlo)

            ce_loss = criterion(logits, label)

            ce_loss_sum += ce_loss.item()

            preds = logits.max(1, keepdim=True)[1]

            correct += preds.eq(label.view_as(preds)).sum().item()

            all_preds.append(preds)
            all_labels.append(label)

    all_preds = torch.cat(all_preds, axis=0)
    all_labels = torch.cat(all_labels, axis=0)

    all_preds, all_labels = all_preds.cpu(), all_labels.cpu()

    # count = [i for i in all_preds]

    # print(Counter(count).keys()) # equals to list(set(words))
    # print(Counter(count).values()) # counts the elements' frequency

    # count = [i for i in all_labels]

    # print(Counter(count).keys()) # equals to list(set(words))
    # print(Counter(count).values()) # counts the elements' frequency

    loss_avg = ce_loss_sum / (i+1)
    metric_loss_avg = metric_loss_sum / (i+1)

    #accuracy
    accuracy = correct / len(loader.dataset)
    #f1 score
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    roc_auc = roc_auc_score(all_labels, all_preds, average='macro') 
    # constant for classes
    # classes = ('Normal', 'Benign','Recall')
    classes = ('Benign','Recall')
    # classes = ('Density A','Density B','Density C','Density D')

    # Build confusion matrix
    cf_matrix = confusion_matrix(all_labels, all_preds)
    cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]

    return loss_avg, metric_loss_avg, accuracy, f1_macro, f1_micro, roc_auc, cmn