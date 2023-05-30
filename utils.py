import config0
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import gc
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
import torch


def train_epoch(network, loader, optimizer, wandb):
    cumu_loss = 0
    
    criterion = nn.CrossEntropyLoss()
    for _, (data, target) in enumerate(loader):
        data, target = data.to(config0.DEVICE), target.to(config0.DEVICE)
        optimizer.zero_grad()

        # ➡ Forward pass
        loss = criterion(network(data), target)
        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)

def train_one_fold(model,
                   optimizer, scheduler, 
                   train_loader, valid_loader, 
                   epochs, device, wandb):
    best_epoch_loss = np.inf
    best_score = 0

    for epoch in range(epochs):
        gc.collect()
        train_epoch_loss, train_score = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=device, epoch=epoch)
        print(f"Train F1 score : {train_score} / Train Loss : {train_epoch_loss}")

        val_epoch_loss, _val_score = valid_one_epoch(model, optimizer,
                                         dataloader = valid_loader, 
                                         device=device, epoch=epoch)
        print(f"Val F1 score : {_val_score} / Val Loss : {val_epoch_loss}")

        if val_epoch_loss <= best_epoch_loss:
            print(f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
        
        if best_score < _val_score:
            best_score = _val_score

        wandb.log({'epochs': epochs})
        wandb.log({'Train F1 score': train_score, 'trainloss': train_epoch_loss})     
        wandb.log({'Val F1 score': _val_score, 'val loss': val_epoch_loss})     


    print("Best val F1 score: {:.4f}".format(best_score))
    print("Best val Loss: {:.4f}".format(best_epoch_loss))


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    
    preds, true_labels = [], []

    losses = AverageMeter()
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype=torch.long)
        
        batch_size = ids.size(0)

        outputs = model(ids, mask)

        loss = criterion(outputs, targets)
        losses.update(loss.item(), outputs.size(0))
        loss.backward()
    
        preds += outputs.argmax(1).detach().cpu().numpy().tolist()
        true_labels += targets.detach().cpu().numpy().tolist()

        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()
        
        bar.set_postfix(Epoch=epoch, Train_Loss=losses.avg,
                        LR=optimizer.param_groups[0]['lr'])
    train_score = f1_score(true_labels, preds, average='weighted')

    gc.collect()
    return losses.avg, train_score

@torch.no_grad()
def valid_one_epoch(model, optimizer, dataloader, device, epoch):
    model.eval()

    preds = []
    true_labels = []
    losses = AverageMeter()
 
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    print(len(dataloader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        targets = data['target'].to(device, dtype=torch.long)
        
        batch_size = ids.size(0)

        outputs = model(ids, mask)

        loss = criterion(outputs, targets)
        losses.update(loss.item(), outputs.size(0))

        preds += outputs.argmax(1).detach().cpu().numpy().tolist()
        true_labels += targets.detach().cpu().numpy().tolist()
      
        bar.set_postfix(Epoch=epoch, Valid_Loss=losses.avg,
                        LR=optimizer.param_groups[0]['lr'])   

    _val_score = f1_score(true_labels, preds, average='weighted')
    
    gc.collect()
    return losses.avg, _val_score

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)