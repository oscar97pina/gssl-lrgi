from typing import List, Optional, Union

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as gnn
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor)
from torch_geometric.utils import spmm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import SparseTensor, matmul

def get_activation(act, **kwargs):
    if act == "relu":
        return nn.ReLU(**kwargs)
    elif act == "elu":
        return nn.ELU(**kwargs)
    elif act == "prelu":
        return nn.PReLU(**kwargs)
    elif act is None:
        return nn.Identity()
        
class FCNN(nn.Sequential):
    def __init__(self, *sizes, act="relu"):
        mls = list()
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            mls.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                mls.append(get_activation(act))
        super().__init__(*mls)

        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

class PropagateStep(gnn.MessagePassing):
    def __init__(self, aggr: Optional[Union[str, List[str], Aggregation]] = "add", 
                       add_self_loops : bool = False,
                       **kwargs):
        norm = False
        if aggr == "norm":
            aggr = "add"
            norm = True
        
        super().__init__(aggr=aggr, **kwargs)

        self.add_self_loops = add_self_loops
        self.norm = norm
    
    def normalize(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None):
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                False, self.add_self_loops, self.flow, x.dtype)

        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim),
                False, self.add_self_loops, self.flow, x.dtype)
        
        return edge_index, edge_weight

    def forward(self, x: Union[Tensor, OptPairTensor], 
                      edge_index: Adj, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        
        edge_weight = None
        if self.norm:
            edge_index, edge_weight = self.normalize(x[0], edge_index)

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, size=size, edge_weight=edge_weight)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        return spmm(adj_t, x[0], reduce=self.aggr)

class Propagate(nn.Module):
    def __init__(self, num_steps, aggr="norm",**kwargs):
        super().__init__()
        self.num_steps = num_steps
        steps = [PropagateStep(aggr=aggr, **kwargs) for i in range(num_steps)]
        self.steps = nn.ModuleList(steps)
        self.aggr  = aggr
    
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj) -> Tensor:
        for step in self.steps:
            x = step(x, edge_index)
        return x
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_steps},{self.aggr})"

class GATBlock(nn.Module):
    def __init__(self, in_size, out_size, concat=True, heads=1, act="elu", **kwargs):
        super().__init__()
        assert out_size % heads == 0
        self.conv = gnn.GATConv(in_size, out_size // heads if concat else out_size, 
                                heads=heads, concat=concat,**kwargs)
        self.skip = nn.Linear(in_size, out_size, bias=False)
        self.act  = get_activation(act)
        self.out_size = out_size
    
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index) + self.skip(x)
        x = self.act(x)
        return x
    
class GATBlockList(nn.Module):
    def __init__(self, convs):
        super().__init__()
        self.convs = nn.ModuleList(convs)
    
    @property
    def out_size(self):
        return self.convs[-1].out_size
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        return x

class GAT(GATBlockList):
    def __init__(self, *sizes, concat=True, heads=1, act="elu", **kwargs):
        convs = list()
        assert len(sizes) >= 2
        if len(sizes) == 2:
            convs.append(GATBlock(*sizes, concat=False, heads=heads, act=act, **kwargs))
        else:
            for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
                convs.append(GATBlock(in_size, out_size, 
                                    concat=concat, heads=heads,
                                    act=act, **kwargs))
        
        super().__init__(convs)

class BigGAT(GATBlockList):
    def __init__(self, *sizes, concat=True, heads=1, act="elu", **kwargs):
        convs = list()
        assert len(sizes) >= 2
        if len(sizes) == 2:
            convs.append(GATBlock(*sizes, concat=False, heads=heads, act=act, **kwargs))
        else:
            # first layer
            convs.append(GATBlock(sizes[0], heads * sizes[1] if concat else sizes[1],
                                concat=concat, heads=heads,
                                act=act, **kwargs))
            for i, (in_size, out_size) in enumerate(zip(sizes[1:-2], sizes[2:-1])):
                in_size  = heads * in_size  if concat else in_size
                out_size = heads * out_size if concat else out_size
                convs.append(GATBlock(in_size, out_size, 
                                    concat=concat, heads=heads,
                                    act=act, **kwargs))
            # last layer
            convs.append(GATBlock(heads * sizes[-2] if concat else sizes[-2], sizes[-1],
                                concat=False, heads=heads,
                                act=act, **kwargs))
        super().__init__(convs)