import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from lrgi.functional import encode, encode_nbatch

def eval_transductive(encoder, data, device, 
                     is_multilabel=False, epochs=100, 
                     grid={'lr': [1e-2], 'wd': [0]}):
    # get representations and labels
    x, y = encode(encoder, [data], device)

    # split
    x_train, y_train = x[data.train_mask], y[data.train_mask]
    x_val,   y_val   = x[data.val_mask],   y[data.val_mask]
    x_test,  y_test  = x[data.test_mask],  y[data.test_mask]
    
    # create data tuples
    train_data = (x_train, y_train)
    val_data   = (x_val,   y_val)
    test_data  = (x_test,  y_test)

    return base_evaluation(train_data, val_data, test_data, 
                           device, is_multilabel=is_multilabel, epochs=epochs,
                           grid = grid)

def eval_inductive_multigraph(encoder, train_loader, val_loader, test_loader, device, 
                              is_multilabel=True, epochs=100, 
                              grid={'lr': [1e-2], 'wd': [0]}):
    # get representations and labels
    x_train, y_train = encode(encoder, train_loader, device)
    x_val, y_val     = encode(encoder, val_loader,   device)
    x_test, y_test   = encode(encoder, test_loader,  device)

    # create data tuples
    train_data = (x_train, y_train)
    val_data   = (x_val,   y_val)
    test_data  = (x_test,  y_test)

    return base_evaluation(train_data, val_data, test_data, 
                           device, is_multilabel=is_multilabel, epochs=epochs, grid=grid)

def eval_inductive(encoder, subdata, data, train_loader, eval_loader, device, 
                  is_multilabel=False, epochs=100, 
                  grid = {'lr': [1e-2], 'wd': [0]}):
    
    # get representations and labels
    x_train, y_train = encode_nbatch(encoder, subdata, train_loader, device)
    x, y = encode_nbatch(encoder, data, eval_loader, device)

    # split
    #x_train, y_train = x[data.train_mask], y[data.train_mask]
    x_val,   y_val   = x[data.val_mask],   y[data.val_mask]
    x_test,  y_test  = x[data.test_mask],  y[data.test_mask]

    # create data tuples
    train_data = (x_train, y_train)
    val_data   = (x_val,   y_val)
    test_data  = (x_test,  y_test)

    fn_eval = base_evaluation if x.shape[0] * x.shape[1] < 2e9 else base_evaluation_batched
    return fn_eval(train_data, val_data, test_data, device, 
                    is_multilabel=is_multilabel, epochs=epochs, grid = grid)\
  
def base_evaluation(train_data, val_data, test_data, device,
                    is_multilabel : bool = False, epochs : int  = 100,
                    grid = {'lr': [1e-2], 'wd': [0]}):
    
    # unpack data
    x_train, y_train = train_data
    x_val,   y_val   = val_data
    x_test,  y_test  = test_data

    # number of classes
    if is_multilabel:
        num_classes = y_train.size(1)
    else:
        num_classes = int(y_train.max()) + 1
    
    # move to device
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val,   y_val   = x_val.to(device), y_val.to(device)
    x_test,  y_test  = x_test.to(device), y_test.to(device)

    # standardize features
    mean, std = x_train.mean(0), x_train.std(0)
    x_train = (x_train - mean) / std
    x_val   = (x_val   - mean) / std
    x_test  = (x_test  - mean) / std
    
    # fit downstream task
    best_val_score, best_test_score = 0, 0
    for lr in grid['lr']:
        for wd in grid['wd']:

            # create logistic regression classifier
            classifier = LogReg(x_train.size(1), num_classes,
                                is_multilabel=is_multilabel).to(device)

            # create optimizer (w/ or w/o weight decay)
            if wd > 0.0:
                optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=wd)
            else:
                optimizer = optim.Adam(classifier.parameters(), lr=lr)

            # train epochs
            for epoch in range(epochs):

                # fit epoch
                optimizer.zero_grad()
                logits = classifier(x_train)
                loss   = classifier.criterion(logits, y_train)
                loss.backward()
                optimizer.step()

                # eval epoch
                val_score  = classifier.evaluate(x_val,  y_val)

                # update best
                if val_score > best_val_score:
                    test_score = classifier.evaluate(x_test, y_test)
                    best_val_score  = val_score
                    best_test_score = test_score
    
    return best_val_score, best_test_score

def base_evaluation_batched(train_data, val_data, test_data, device,
                            is_multilabel : bool = False, epochs : int  = 100,
                            grid = {'lr': [1e-2], 'wd': [0]}):
    
    # unpack data
    x_train, y_train = train_data
    x_val,   y_val   = val_data
    x_test,  y_test  = test_data

    # standardize
    mean, std = x_train.mean(0), x_train.std(0)
    x_train = (x_train - mean) / std
    x_val   = (x_val   - mean) / std
    x_test  = (x_test  - mean) / std

    # number of classes
    if is_multilabel:
        num_classes = y_train.size(1)
    else:
        num_classes = int(y_train.max()) + 1

    # create pytorch tensor datasets
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset   = torch.utils.data.TensorDataset(x_val,   y_val)
    test_dataset  = torch.utils.data.TensorDataset(x_test,  y_test)

    # create pytorch dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=64, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=64, shuffle=False)

    # fit downstream task
    best_val_score, best_test_score = 0, 0
    for lr in grid['lr']:
        for wd in grid['wd']:

            # create logistic regression classifier
            classifier = LogReg(x_train.size(1), num_classes,
                                is_multilabel=is_multilabel,
                                batched=True).to(device)

            # create optimizer (w/ or w/o weight decay)
            if wd > 0.0:
                optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=wd)
            else:
                optimizer = optim.Adam(classifier.parameters(), lr=lr)

            # train epochs
            for epoch in range(epochs):
                # fit epoch
                # as opposed to standard pytorch training, 
                # we update for the entire epoch, not per batch
                # to replicate the behavior of base_evaluation
                optimizer.zero_grad()
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = classifier(xb)
                    loss   = classifier.criterion(logits, yb) / len(train_dataset)
                    loss.backward()
                optimizer.step()

                # eval epoch
                val_score  = classifier.evaluate_batch(val_loader, device)

                # update best
                if val_score > best_val_score:
                    test_score = classifier.evaluate_batch(test_loader, device)
                    best_val_score  = val_score
                    best_test_score = test_score
    
    return best_val_score, best_test_score
    

class LogReg(nn.Module):
    def __init__(self, in_size, num_classes, is_multilabel=False, batched=False):
        super().__init__()
        self.linear = nn.Linear(in_size, num_classes)
        self.is_multilabel = is_multilabel
        reduction = 'sum' if batched else 'mean'
        self._criterion = nn.BCEWithLogitsLoss(reduction=reduction) if is_multilabel else nn.CrossEntropyLoss(reduction=reduction)
    
    def criterion(self, logits, target):
        if self.is_multilabel:
            return self._criterion(logits, target.float())
        else:
            return self._criterion(logits, target)
        
    def forward(self, x):
        return self.linear(x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self(x)
        if self.is_multilabel:
            return (logits > 0).float().cpu()
        else:
            return logits.argmax(dim=-1).cpu()
    
    def predict_batch(self, loader, device):
        self.eval()
        with torch.no_grad():
            logits = []
            for batch in loader:
                x_batch, _ = batch
                logits.append(self(x_batch.to(device)).cpu())
            logits = torch.cat(logits, dim=0)
        if self.is_multilabel:
            return (logits > 0).float()
        else:
            return logits.argmax(dim=-1)
    
    def evaluate(self, x, y):
        pred = self.predict(x)
        if self.is_multilabel:
            return metrics.f1_score(y.cpu(), pred, average='micro') if pred.sum() > 0 else 0
        else:
            return metrics.accuracy_score(y.cpu(), pred)
    
    def evaluate_batch(self, loader, device):
        target = torch.cat([y for _, y in loader], dim=0).cpu()
        pred = self.predict_batch(loader, device)
        if self.is_multilabel:
            return metrics.f1_score(target, pred, average='micro') if pred.sum() > 0 else 0
        else:
            return metrics.accuracy_score(target, pred)