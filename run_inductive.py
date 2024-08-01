import os
import copy
import argparse
from sklearn.utils import Bunch
import wandb

import torch
import torch.nn as nn

from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T

from lrgi.data import get_ogbn, get_reddit
from lrgi.models import GATBlockList, GAT, FCNN, Propagate
from lrgi.rgi import RGI
from lrgi.utils import get_device, get_optimizer, get_scheduler, update_lr, set_seed, save_ckpt
from lrgi.functional import fit_rgi, encode_nbatch
from lrgi.logreg import eval_inductive

import wandb

def load_data(args):
    # load data
    if args.dataset == "reddit":
        data = get_reddit(args.dataset_dir, transform=None)
    elif args.dataset in ["products"]:
        print(f"Loading {args.dataset} from {args.dataset_dir}")
        data = get_ogbn(args.dataset_dir, args.dataset, transform=None)
        print(data)
    # get train subgraph
    subdata = data.subgraph(data.train_mask)
    print(subdata)
    
    return data, subdata

def create_loaders(args, data, subdata):
    # load data
    if args.method == "lrgi_no_samp":
        num_neighbors = [-1] + [5] * args.K
    elif args.dataset == "reddit":
        num_neighbors = ([10] if args.method == "lrgi" else [10] * (args.num_layers)) + [5] * (args.K)
    elif args.dataset == "products":
        #num_neighbors = [10] * (args.num_layers - 1 if args.method == "rgi" else 1) + [5] * (args.K)
        num_neighbors = ([10] if args.method == "lrgi" else [10] * (args.num_layers)) + [5] * (args.K)

    print(num_neighbors)
    
    # data loader to train the network
    train_loader = NeighborLoader(copy.deepcopy(subdata), input_nodes=None,
                                num_neighbors=num_neighbors, batch_size=args.batch_size,
                                shuffle=True, num_workers=args.num_workers)

    # data loaders for evaluation
    train_subgraph_loader = NeighborLoader(copy.deepcopy(subdata), input_nodes=None,
                                    num_neighbors=[-1], batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers)

    eval_subgraph_loader = NeighborLoader(copy.deepcopy(data), input_nodes=None,
                                    num_neighbors=[-1], batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers)

    # No need to maintain these features during evaluation:
    del train_subgraph_loader.data.x, train_subgraph_loader.data.y
    del train_subgraph_loader.data.train_mask, train_subgraph_loader.data.val_mask, train_subgraph_loader.data.test_mask
    # Add global node index information.
    train_subgraph_loader.data.num_nodes = subdata.num_nodes
    train_subgraph_loader.data.n_id = torch.arange(subdata.num_nodes)

    # No need to maintain these features during evaluation:
    del eval_subgraph_loader.data.x, eval_subgraph_loader.data.y
    del eval_subgraph_loader.data.train_mask, eval_subgraph_loader.data.val_mask, eval_subgraph_loader.data.test_mask
    # Add global node index information.
    eval_subgraph_loader.data.num_nodes = data.num_nodes
    eval_subgraph_loader.data.n_id = torch.arange(data.num_nodes)

    return train_loader, train_subgraph_loader, eval_subgraph_loader

def run(args):
    import time
    # wandb
    wandb.init(project=args.project, config=args)

    # device
    device = get_device()

    # seed
    set_seed(args.seed)

    # load data
    data, subdata = load_data(args)

    # create loaders
    train_loader, train_subgraph_loader, eval_subgraph_loader = create_loaders(args, data, subdata)

    # build network
    in_size  = data.x.size(1)
    sizes    = [in_size] + (args.num_layers) * [args.emb_size]
    encoder  = GAT(*sizes, heads=args.heads, concat=True, dropout=0.0)
    print(encoder)
    
    # arguments
    rgi_kwargs = {
        "epochs"  : args.epochs,
        "lr"      : args.lr,
        "wd"      : args.wd,
        "lr_warmup_epochs": args.lr_warmup_epochs,
        "shift"   : args.shift,
        "K"       : args.K,
        "hid"     : args.hid,
        "p_drop_x": args.p_drop_x,
        "p_drop_u": args.p_drop_u,
        "lambda_1": args.lambda_1,
        "lambda_2": args.lambda_2,
        "lambda_3": args.lambda_3,}
    # pretrain
    t0 = time.time()
    if args.method is None:
        pass
    elif args.method == "rgi":
        # pretrain the network end2end with RGI
        encoder = fit_rgi(encoder, train_loader, device, 
                        **rgi_kwargs)
    elif args.method.startswith("lrgi"):
        # pretrain the network layer by layer with LRGI
        for i in range(len(encoder.convs)):
            # pretrain the i-th layer
            encoder.convs[i] = fit_rgi(encoder.convs[i], train_loader, device, 
                                       **rgi_kwargs)
            # update the train_loader data (inductive)
            tmp_encoder = GATBlockList( [copy.deepcopy(encoder.convs[i].cpu())] )
            x_train, y_train = encode_nbatch(tmp_encoder, train_loader.data, 
                                             train_subgraph_loader, device)
            train_loader.data.x = x_train
            train_loader.data.y = y_train

    print("Pretraining time: {:.3f}".format(time.time() - t0))
    # evaluate
    val_f, test_f = eval_inductive(encoder, subdata, data, 
                                    train_subgraph_loader, 
                                    eval_subgraph_loader, device,
                                    is_multilabel=False,
                                    epochs=1000, 
                                    grid = {'lr': [1e-2], 'wd': [0]})

    wandb.log({"val_f": val_f, "test_f": test_f, "layer": args.num_layers, "epoch": args.epochs+1})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inductive experiments')
    parser.add_argument("--project", type=str,  default="gssl_test")
    parser.add_argument("--experiment", type=str,  default="test")
    parser.add_argument("--method",  type=str,  default="rgi")
    
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset", type=str,      default="products")
    parser.add_argument("--dataset_dir", type=str,  default="./")
    parser.add_argument("--results_dir", type=str,  default=None)

    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)

    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--shift", type=str, default="mean")
    
    parser.add_argument("--hid", type=int, default=8)

    parser.add_argument("--lambda_1", type=float, default=25)
    parser.add_argument("--lambda_2", type=float, default=25)
    parser.add_argument("--lambda_3", type=float, default=1)

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float,  default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    
    parser.add_argument("--lr_warmup_epochs", type=int, default=None)

    parser.add_argument("--p_drop_x",  type=float, default=0.0)
    parser.add_argument("--p_drop_u", type=float, default=0.0)

    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()
    from sklearn.utils import Bunch
    args = Bunch(**args.__dict__)
    
    run(args)
  