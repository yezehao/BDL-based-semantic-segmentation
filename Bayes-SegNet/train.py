# basic modules
import os 
import json
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from torchmetrics import Accuracy, JaccardIndex
# SegNet Model with Bayes / None-Bayes
import segnet
# DataLoader modules
from MaSTr1325 import MaSTr1325
# Pytorch modules
import torch
from torch import nn
from torch.optim import AdamW, SGD
from torchvision import transforms
from torch.utils.data import DataLoader
# selfdefined utils modules
from utils import PILToLongTensor, to_numpy, median_freq_balancing

torch.autograd.set_detect_anomaly(True) 

def train_step(model, dataloader, optimizer, criterion, device):
    logs = []
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y.to(device).squeeze(1) # targets only have one channel
        optimizer.zero_grad()
        y_logit = model(X)
        y_logprob = torch.log_softmax(y_logit, dim=1)  # Apply log softmax # New Added
        pred = torch.argmax(y_logprob, 1)
        acc = pred.eq(y.data.view_as(pred)).float().cpu().mean()
        loss = criterion(y_logprob, y)
        loss.backward()
        optimizer.step()
        logs.append([to_numpy(loss), to_numpy(acc)])
    return np.array(logs)

def evaluate(model, dataloader, metrics, device, root, k=10, use_dropout=True):
    if use_dropout:
        model.train()
    else:
        model.eval()
        
    for metric in metrics:
        metric.reset()
    
    with torch.no_grad():
        # Initialization
        i = 0
        threshold = 0.2
        Prs, Res, F1 = 0, 0, 0
        
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).squeeze(1)
            
            if k > 1:
                # monte carlo samples
                y_logit = []
                for _ in range(k):
                    y_logit.append(model(X))
                y_logit = torch.stack(y_logit, dim=0).squeeze(0)
                y_logit = y_logit.mean(0)
            else:
                y_logit = model(X)

            predict = y_logit[0]
            y_value, prediction = torch.max(predict, dim=0)
            prediction[y_value < threshold] = 4

            GT_path = os.path.join(root, f'val_mask/{i+1:03}m.png') # Groud Truth
            GT_mask = Image.open(GT_path)
            transform = transforms.ToTensor()
            GT = transform(GT_mask).to(device)

            # True Positives (TPs)
            TP =  (GT == 0) & (prediction == 0)
            TPs = TP.sum().item()
            # False Positives (FPs)
            FP = (GT != 0) & (prediction == 0)
            FPs = FP.sum().item()
            # False Negatives (FNs)
            FN = (GT == 0) & (prediction != 0)
            FNs = FN.sum().item()

            Pr = TPs/(TPs + FPs) # Precision (Pr)
            Re = TPs/(TPs + FNs) # Recall (Re)
            if (Pr+Re)==0:
                F1 = 0
            else:
                F1 += 2*Pr*Re/(Pr + Re) # Harmonic Mean F1
            Prs += Pr
            Res += Re
            
            # Next Image
            i += 1
            
            for metric in metrics:
                metric.update(y_logit, y)

    # Mean
    Prs = Prs/(len(dataloader))
    Res = Res/(len(dataloader))
    F1 = F1/(len(dataloader))
    print(f"Precision: {Prs}, Recall: {Res}, F1: {F1}")

    evaluation = {
        'Precision': Prs,
        'Recall': Res,
        'F1': F1
    }

    # with open('eval.txt', 'a') as file:
    #     file.write(f"Precision: {Prs}, Recall: {Res}, F1: {F1}\n")

    
    # FIX: torchmetrics do no rule out ignored index
    log = []
    for metric in metrics:
        vals = metric.compute()
        if metric.ignore_index != None:
            vals[metric.ignore_index] = torch.nan
        val = torch.nanmean(vals)
        log.append(to_numpy(val))


    return np.array(log), evaluation

def main(args):
    lr = 1e-3
    weight_decay = 5e-4
    momentum = 0.9
    epochs = args.epoch
    batch_size = 8
    best_acc = 0
    device = args.device
    # Dataset Selection
    if args.data_path == 'MaSTr1325':
        # Define root path
        # root = os.path.expanduser("~/autodl-tmp/MaSTr1325") # this need to be modified
        root = os.path.expanduser("~/BDL/BDL-based-semantic-segmentation/Dataset/MaSTr1325")
        print(f"root path: {root}")
        # Define transform
        transform = transforms.Compose([
            transforms.Resize((384, 512)),
            transforms.ToTensor(),
            # LocalContrastNormalisation(3, 0, 1)
        ])
        target_transform = transforms.Compose([
            transforms.Resize((384, 512)),
            PILToLongTensor()
        ])
        # DataLoader for train/val/test
        train_data = MaSTr1325(root, "train", transform=transform, target_transform=target_transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_data = MaSTr1325(root, "val", transform=transform, target_transform=target_transform)
        val_loader = DataLoader(val_data, batch_size=1)
        test_data = MaSTr1325(root, "test", transform=transform, target_transform=target_transform)
        test_loader = DataLoader(test_data, batch_size=1)

    num_classes = len(train_data.color_encoding)
    print(f"The Number of Classes: {num_classes}")

    # # weights are copied from https://github.com/alexgkendall/SegNet-Tutorial
    # class_weights = torch.tensor(
    #     [0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 
    #      9.6446, 1.8418, 0.6823, 6.2478, 7.3614, 0.0], 
    #     device=device)
    class_weights = median_freq_balancing(train_loader, num_classes, device=device)
    print(f"Weight before ignore unlabelling: {class_weights}")
    # class_weights[-1] = 0.0
    
    model = segnet.BayesSegNet(in_channels=3, out_channels=num_classes, vgg_encoder=True)
    model.to(device)
    # print(model)
    # optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.NLLLoss(weight=class_weights, ignore_index=3)
    metrics = [
        Accuracy(task="multiclass", num_classes=num_classes, ignore_index=num_classes-1, average="none").to(device),
        JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=num_classes-1, average="none").to(device),
    ]
    train_logs, val_logs, evaluations = [], [], []
    for i in range(epochs):
        train_log = train_step(model, train_loader, optimizer, criterion, device)
        print("Epoch {}, last mini-batch nll={}, acc={}".format(i+1, train_log[-1][0], train_log[-1][1]))
        train_logs.append(train_log)
        
        val_log, evaluation = evaluate(model, val_loader, metrics, device, root, k=10, use_dropout=True)
        print("Epoch {}, val acc={}, iou={}".format(i+1, val_log[0], val_log[1]))
        val_logs.append(val_log[np.newaxis])
        evaluations.append(evaluation)
        
        # save best model
        if val_log[0] > best_acc:
            best_acc = val_log[0]
            best_epoch = i + 1
            best_model = deepcopy(model.state_dict())
            torch.save(best_model, r'./saved_model/SegNet_'+str(args.data_path)+'_'+str(batch_size)+'_'+str(best_epoch)+'.pth')
    
    train_logs, val_logs = np.concatenate(train_logs, axis=0), np.concatenate(val_logs, axis=0)
    
    model.load_state_dict(best_model)
    test_log,_ = evaluate(model, test_loader, metrics, device, root, k=50, use_dropout=True)
    print("Epoch {}, test acc={}, iou={}".format(best_epoch, test_log[0], test_log[1]))

    # Plot Precision, Recall, and F1 score over epochs
    precisions = [res['Precision'] for res in evaluations]
    recalls = [res['Recall'] for res in evaluations]
    f1_scores = [res['F1'] for res in evaluations]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.plot(np.arange(epochs), precisions, 'b-', label='Precision')
    ax2.plot(np.arange(epochs), recalls, 'r-', label='Recall')
    ax3.plot(np.arange(epochs), f1_scores, 'm-', label='F1 Score')

    for ax, metric in zip([ax1, ax2, ax3], ['Precision', 'Recall', 'F1 Score']):
        ax.legend()
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)

    plt.savefig(f"figures/evaluation_precisions_over_epochs")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="MaSTr1325", help="data directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--epoch", type=int, default=1000, help="epoch number")
    args = parser.parse_args()
    main(args)
