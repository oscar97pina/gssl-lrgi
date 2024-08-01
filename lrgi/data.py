import torch_geometric.datasets as D
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import index_to_mask
from ogb.nodeproppred import PygNodePropPredDataset

class ConcatDataset(InMemoryDataset):
    r"""
    PyG Dataset class for merging multiple Dataset objects into one.
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        self.__indices__ = None
        self.__data_list__ = []
        for dataset in datasets:
            self.__data_list__.extend(list(dataset))
        self.data, self.slices = self.collate(self.__data_list__)

def get_ppi(root, transform=None):
    train_dataset = D.PPI(root, split="train", transform=transform)
    val_dataset   = D.PPI(root, split="val",   transform=transform)
    test_dataset  = D.PPI(root, split="test",  transform=transform)
    return train_dataset, val_dataset, test_dataset

def get_planetoid(root, name, transform=T.NormalizeFeatures()):
    assert name in ['cora','citeseer','pubmed']
    # get dataset
    dataset = D.Planetoid(root, name=name, transform=transform)
    # get data
    data = dataset[0]
    # planetoid includes masks
    return data

def get_ogbn(root, name, transform=None):
    assert name in ['arxiv','products','proteins','papers100M']
    # get dataset
    name = 'ogbn-' + name
    dataset = PygNodePropPredDataset(name=name, root=root, transform=transform)
    # get data
    data = dataset[0]
    # split
    split_idx = dataset.get_idx_split()
    data.train_mask  = index_to_mask(split_idx["train"], size=data.num_nodes)
    data.val_mask    = index_to_mask(split_idx["valid"], size=data.num_nodes)
    data.test_mask   = index_to_mask(split_idx["test"],  size=data.num_nodes)
    # squeeze
    data.y = data.y.squeeze()

    return data

def get_reddit(root, transform=None):
    # get dataset
    dataset = D.Reddit(root, transform=transform)
    # get data
    data = dataset[0]
    # reddit includes masks

    return data