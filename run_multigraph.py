import os
import copy
import argparse
from sklearn.utils import Bunch
import wandb

import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader

from lrgi.data import get_ppi, ConcatDataset
from lrgi.models import GATBlockList, GAT, FCNN, Propagate
from lrgi.rgi import RGI
from lrgi.utils import get_device, get_optimizer, get_scheduler, update_lr, set_seed, save_ckpt
from lrgi.functional import fit_rgi, encode_lsdata
from lrgi.logreg import eval_inductive_multigraph

import wandb

def run(args):
    wandb.init(project=args.project, config=args)

    # device
    device = device = get_device()

    # seed
    set_seed(args.seed)

    # load data
    train_dataset, val_dataset, test_dataset = get_ppi(args.dataset_dir)

    # data loader
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, num_workers=args.num_workers)

    # build network
    in_size  = train_dataset[0].x.size(1)
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
    if args.method is None:
        pass
    elif args.method == "rgi":
        # create cat loader
        cat_loader   = DataLoader(ConcatDataset([train_dataset, val_dataset]), 
                        batch_size=1, shuffle=True, num_workers=args.num_workers)
    
        # pretrain the network end2end with RGI
        encoder = fit_rgi(encoder, cat_loader, device, 
                        **rgi_kwargs)
    elif args.method == "lrgi":

        # list of data objects to train each layer in a SSL fashion
        # at first layer is the concatenation of train and val datasets
        ls_data = [data for data in train_dataset] + \
                  [data for data in val_dataset]

        # pretrain the network layer by layer with LRGI
        for i in range(len(encoder.convs)):

            # create data loader
            lsdata_loader = DataLoader(ls_data, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)
            # pretrain the i-th layer
            encoder.convs[i] = fit_rgi(encoder.convs[i], lsdata_loader, device, 
                                       **rgi_kwargs)

            # update the train_loader data (inductive)
            tmp_encoder = copy.deepcopy(encoder.convs[i].cpu())
            ls_data     = encode_lsdata(tmp_encoder, ls_data, device)
    
    # evaluate
    val_f, test_f = eval_inductive_multigraph(encoder, train_loader, val_loader, test_loader,
                                                device, epochs=1000, is_multilabel=args.dataset == 'ppi',
                                                grid={'lr':[1e-2], 'wd':[0.0]})
    wandb.log({"val_f": val_f, "test_f": test_f, "layer": args.num_layers, "epoch": args.epochs+1})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run PPI experiments')
    parser.add_argument("--project", type=str,  default="(l)rgi")
    parser.add_argument("--experiment", type=str,  default="test")
    parser.add_argument("--method",  type=str,  default="rgi")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset", type=str,      default="ppi")
    parser.add_argument("--dataset_dir", type=str,  default="./")
    parser.add_argument("--results_dir", type=str,  default=None)

    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=4)

    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--shift", type=str, default="mean")

    parser.add_argument("--hid", type=int, default=8)

    parser.add_argument("--lambda_1", type=float, default=15)
    parser.add_argument("--lambda_2", type=float, default=10)
    parser.add_argument("--lambda_3", type=float, default=10)

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float,  default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr_warmup_epochs", type=int, default=-1)

    parser.add_argument("--p_drop_x",  type=float, default=0.0)
    parser.add_argument("--p_drop_u", type=float, default=0.0)

    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()
    from sklearn.utils import Bunch
    args = Bunch(**args.__dict__)

    run(args)