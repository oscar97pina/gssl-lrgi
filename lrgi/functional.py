import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

import wandb

from .models import FCNN, Propagate
from .rgi import RGI
from .utils import get_optimizer, get_scheduler, update_lr

##########################################################
# Train RGI
##########################################################
def fit_rgi_epoch(rgi, loader, optimizer, device):
    """
    Train lrgi with data from loader
    """
    rgi.train()
    ls_logs = list()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss, logs = rgi.loss(data)
        loss.backward()
        optimizer.step()
        ls_logs.append(logs)
    # average logs
    logs = {k: sum([logs[k] for logs in ls_logs]) / len(ls_logs) for k in ls_logs[0]}
    return logs

def fit_rgi(encoder, loader, device,
    shift="mean", K=1, hid=8,
    p_drop_x=0.0, p_drop_u=0.0, 
    lambda_1=1, lambda_2=1, lambda_3=1,
    epochs=100, lr=1e-4, wd=1e-5, lr_warmup_epochs=10):

    # get propagation function
    propagate = Propagate(K, shift)

    # create reconstruction networks
    emb_size = encoder.out_size
    local_pred  = FCNN(emb_size, hid * emb_size, emb_size)
    global_pred = FCNN(emb_size, hid * emb_size, emb_size)

    # create RGI
    rgi = RGI(encoder, propagate, local_pred, global_pred,
              p_drop_x, p_drop_u, lambda_1, lambda_2, lambda_3)
    rgi.to(device)

    # optimizer and scheduler
    optimizer = get_optimizer(rgi.parameters(), lr=lr, wd=wd)
    scheduler = get_scheduler(lr, lr_warmup_epochs, epochs)

    # train rgi
    for epoch in range(1, epochs+1):
        # update learning rate
        update_lr(epoch, scheduler, optimizer)

        # fit epoch
        logs = fit_rgi_epoch(rgi, loader, optimizer, device)
        logs.update({"epoch": epoch})
        wandb.log(logs)
    
    return rgi.local_nn

##########################################################
# Get embeddings and labels
##########################################################
def encode(encoder, loader, device):
    """
    Encode data from loader with encoder
    """
    encoder = encoder.to(device)
    encoder.eval()
    xs, ys = list(), list()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x = data.x
            edge_index = data.edge_index if data.edge_index is not None else data.adj_t
            out = encoder(x, edge_index)
            xs.append(out.cpu())
            ys.append(data.y.cpu())
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return x, y

def encode_nbatch(encoder, data, loader, device):
    encoder.eval()
    encoder.to(device)
    x_all = data.x
    y_all = data.y
    with torch.no_grad():
        for i, conv in enumerate(encoder.convs):
            xs, ys = list(), list()
            for nbatch in loader:
                # get x from node ids
                x = x_all[nbatch.n_id].to(device)
                # get edge_index and edge_attr (if exists)
                edge_index = nbatch.edge_index.to(device)
                # forward through layer
                x = conv(x, edge_index)
                # get the first batch_size nodes
                xs.append(x[:nbatch.batch_size].cpu())
                ys.append(y_all[nbatch.n_id[:nbatch.batch_size]].cpu())
            # cat x for next layer
            x_all = torch.cat(xs, dim=0)
            y_all = torch.cat(ys, dim=0)

    return x_all, y_all

def encode_lsdata(encoder, ls_data, device):
    """
    Encode data from ls_data with encoder.
    This is used for intermediate representation.
    """
    encoder = encoder.to(device)
    encoder.eval()
    enc_ls_data = list()
    with torch.no_grad():
        for data in ls_data:
            data = data.to(device)
            x    = encoder(data.x, data.edge_index)
            data.x = x.detach()
            enc_ls_data.append(data.cpu())
    return enc_ls_data