from sklearn import metrics
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils.dropout import dropout_edge

from typing import List, Optional, Tuple, Union
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor
)

def fn_dropout(x: Tensor, p: float):
    if p > 0.0:
        return F.dropout(x, p=p)
    return x

def fn_dropedge(edge_index: Adj, p: float, num_nodes: int):
    if p > 0.0:
        if isinstance(edge_index, Tensor):
            edge_index, mask = dropout_edge(edge_index, p=p)

        elif isinstance(edge_index, SparseTensor):
            row, col, _ = edge_index.t().coo()
            edge_index  = torch.stack([row, col], dim=0)
            edge_index  = fn_dropedge(edge_index, p, num_nodes)
            row, col = edge_index
            edge_index = SparseTensor(row=col, col=row,
                          sparse_sizes=(num_nodes, num_nodes))
    
    return edge_index

def off_diagonal(x: Tensor):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def entropy_loss(x: Tensor):
    B, D = x.size(0), x.size(1)
    # normalize
    x_norm = x - x.mean(dim=0)
    # covariance matrix
    cov_x = (x_norm.T @ x_norm) / (B - 1)
    # variance loss
    var_loss = torch.diagonal(cov_x).add_(-1).pow_(2).sum().div(D)
    # covariance
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(D)

    return var_loss, cov_loss

class RGI(nn.Module):
    def __init__(self, local_nn, global_nn,
                       local_pred, global_pred,
                       p_drop_x  = 0.0,
                       p_drop_u  = 0.0,
                       lambda_1=1, lambda_2=1, lambda_3=1):
        super().__init__()
        # encoders
        self.local_nn  = local_nn
        self.global_nn = global_nn

        # predictors
        self.local_pred  = local_pred
        self.global_pred = global_pred

        # lambdas
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

        # dropout - dropedge
        self.p_drop_x = p_drop_x
        self.p_drop_u = p_drop_u
    
    def forward(self, x, edge_index, edge_attr=None):
        return self.local_nn(x, edge_index, edge_attr=edge_attr)
    
    def trainable_parameters(self):
        return list(self.local_nn.parameters())  + list(self.global_nn.parameters()) \
             + list(self.local_pred.parameters()) + list(self.global_pred.parameters())
    
    def local_loss(self, u, v):
        u_pred = self.local_pred(v)
        rec_loss = F.mse_loss(u, u_pred)
        var_loss, cov_loss = entropy_loss(u)
        return rec_loss, var_loss, cov_loss
    
    def global_loss(self, u, v):
        v_pred = self.global_pred(u)
        rec_loss = F.mse_loss(v, v_pred)
        var_loss, cov_loss = entropy_loss(v)
        return rec_loss, var_loss, cov_loss
    
    def total_loss(self, u, v):
        # local and global losses
        local_rec_loss, local_var_loss, local_cov_loss    = self.local_loss(u,v)
        global_rec_loss, global_var_loss, global_cov_loss = self.global_loss(u,v)

        # total loss
        rec_loss = local_rec_loss + global_rec_loss
        var_loss = local_var_loss + global_var_loss
        cov_loss = local_cov_loss + global_cov_loss

        loss = self.lambda_1 * rec_loss +\
               self.lambda_2 * var_loss +\
               self.lambda_3 * cov_loss
        
        logs = dict(loss            = loss.item(),
                    rec_loss        = rec_loss.item(),
                    var_loss        = var_loss.item(),
                    cov_loss        = cov_loss.item(),
                    local_rec_loss  = local_rec_loss.item(),
                    local_var_loss  = local_var_loss.item(),
                    local_cov_loss  = local_cov_loss.item(),
                    global_rec_loss = global_rec_loss.item(),
                    global_var_loss = global_var_loss.item(),
                    global_cov_loss = global_cov_loss.item())        
        
        return loss, logs

    def loss(self, data):
        x = data.x
        edge_index = data.edge_index if data.edge_index is not None else data.adj_t
        num_nodes = x.size(0)
        
        # input dropout
        _x = fn_dropout(x, self.p_drop_x)
        _edge_index = fn_dropedge(edge_index, self.p_drop_x, num_nodes)       

        # local representations
        u = self.local_nn(_x, _edge_index)

        # local dropout
        _u = fn_dropout(u, self.p_drop_u)
        _edge_index  = fn_dropedge(edge_index, self.p_drop_u, num_nodes)  

        # global representations
        v = self.global_nn(_u, _edge_index)

        # if node batched setting, get target nodes
        if 'batch_size' in data:
            batch_size = data.batch_size
            u = u[:batch_size]
            v = v[:batch_size]

        # loss
        loss, logs = self.total_loss(u, v)

        return loss, logs