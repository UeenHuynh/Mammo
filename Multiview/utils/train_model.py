import os
import torch
from tqdm import tqdm
from utils.eval_model import eval
from torch.autograd import Variable
from utils.mixup_utils import mixup_data, mixup_criterion
import matplotlib.pyplot as plt
import seaborn as sn

def train(model,
          device,
          trainloader,
        #   valloader,
          testloader,
          metric_loss,
          miner,
          criterion,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          best_f1):
    best_f1 = best_f1

    # constant for classes
    # classes = ('Normal', 'Benign','Recall')
    classes = ('Benign','Recall')
    # classes = ('Density A','Density B','Density C','Density D')

    for epoch in range(start_epoch + 1, end_epoch + 1):
        f = open(os.path.join(save_path, 'log.txt'), 'a')
        model.train()
        print('Training %d epoch' % epoch)

        lr = next(iter(optimizer.param_groups))['lr']
        for _, data in enumerate(tqdm(trainloader)):
            img_cc, img_mlo, label = data
            img_cc, img_mlo = img_cc.to(device), img_mlo.to(device)
            label = (label-1).to(device)

            optimizer.zero_grad()

            logits = model(img_cc, img_mlo)

            ce_loss = criterion(logits, label) 

            ce_loss.backward()
            
            optimizer.step()
            
        scheduler.step()
        
        f.write('\nEPOCH' + str(epoch) + '\n')
        # eval valset
        # val_loss_avg, val_metric_loss_avg, val_accuracy = eval(model, device, valloader, metric_loss, miner, criterion, split='val')
        # print('Validation set: Avg Val CE Loss: {:.4f}; Avg Val Metric Loss: {:.4f}; Val accuracy: {:.2f}%'.format(val_loss_avg, val_metric_loss_avg, 100. * val_accuracy))
        # f.write('Validation set: Avg Val CE Loss: {:.4f}; Avg Val Metric Loss: {:.4f}; Val accuracy: {:.2f}% \n'.format(val_loss_avg, val_metric_loss_avg, 100. * val_accuracy))
        # eval testset
        test_loss_avg, test_metric_loss_avg, test_accuracy, f1_macro, f1_micro, roc_auc, cmn = eval(model, device, testloader, metric_loss, miner, criterion, split='test')
        print('Test set: Avg Test CE Loss: {:.4f}; Avg Test Metric Loss: {:.4f}; Test accuracy: {:.2f}%; F1 Score_Macro: {:.4f}; F1 Score_Micro: {:.4f}, ROC_AUC: {:.4f}.\t'.format(test_loss_avg, test_metric_loss_avg, 100. * test_accuracy, f1_macro, f1_micro, roc_auc))
        f.write('Test set: Avg Test CE Loss: {:.4f}; Avg Test Metric Loss: {:.4f}; Test accuracy: {:.2f}%; F1 Score_Macro: {:.4f}; F1 Score_Micro: {:.4f}; ROC_AUC: {:.4f}.\t'.format(test_loss_avg, test_metric_loss_avg, 100. * test_accuracy, f1_macro, f1_micro, roc_auc))
        print(cmn)
        f.write(str(cmn))
        # save checkpoint
        print('Saving checkpoint')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'learning_rate': lr,
            'f1_score_macro': f1_macro,
            'f1_score_micro': f1_micro,
            'roc_auc': roc_auc,
            # 'val_acc': val_accuracy,
            'test_acc': test_accuracy
        }, os.path.join(save_path, 'current_model' + '.pth'))

        if f1_macro > best_f1:
            print('Saving best model')

            plt.figure(figsize = (12,7))
            matrix = sn.heatmap(cmn, annot=True, fmt='.2f', xticklabels=[i for i in classes], yticklabels=[i for i in classes])
            plt.title('Confusion Matrix') 
            plt.ylabel('Actal Values')
            plt.xlabel('Predicted Values')
            plt.savefig(rf"{save_path}/cfmatrix_{epoch}.png")

            f.write('\nSaving best model!\n')
            best_f1 = f1_macro
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'learning_rate': lr,
                'f1_score_macro': f1_macro,
                'f1_score_micro': f1_micro,
                'roc_score': roc_auc,
                # 'val_acc': val_accuracy,
                'test_acc': test_accuracy
            }, os.path.join(save_path, 'best_model' + '.pth'))
        f.close()