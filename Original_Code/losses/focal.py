import torch 
import torch.nn as nn 
import torch.nn.functional as F

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, alpha , gamma = 5, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction = self.reduction, weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (((1-pt) ** self.gamma) * ce_loss).mean()

        return focal_loss

