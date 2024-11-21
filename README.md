# LRGI: Layer-wise Training of Graph Neural Networks with Self-Supervised Learning

## Abstract

End-to-end training of graph neural networks (GNN) on large graphs presents several memory and computational challenges, and limits the application to shallow architectures as depth exponentially increases the memory and space complexities. In this manuscript, we propose Layer-wise Regularized Graph Infomax, an algorithm to train GNNs layer by layer in a self-supervised manner. We decouple the feature propagation and feature transformation carried out by GNNs to learn node representations in order to derive a loss function based on the prediction of future inputs. We evaluate the algorithm in inductive large graphs and show similar performance to other end to end methods and a substantially increased efficiency, which enables the training of more sophisticated models in one single device. We also show that our algorithm avoids the oversmoothing of the representations, another common challenge of deep GNNs.

## Installation

1. Create a Virtual environment
```bash
python3 -m venv myenv
```

2. Activate the environment
```bash
source myenv/bin/activate
```

3. Install torch==2.3.0. We've used cuda 11.8. You can find more information in the [official website](https://pytorch.org/get-started/locally/).
```bash
pip3 install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

4. Install torch_geometric and other optional packages:
```bash
pip3 install torch_geometric

pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
```

5. Install the other requirements
```bash
pip3 install -r requirements.txt
```


## Usage

### Project Structure
```bash
gssl-lrgi/
|-- lrgi/           # main package
|   |-- data.py     # functions to download and load data
|   |-- functional.py # functions to train the model and encode data
|   |-- logreg.py   # logistic regression evaluation functions
|   |-- models.py   # graph neural networks and fully-connected models
|   |-- rgi.py      # implementation of RGI model
|   |-- scheduler.py # learning rate scheduler
|   |-- utils.py    # other utilities
|-- run_inductive.py # training file for one graph datasets (Reddit and ogbn-products)
|-- run_multigraph.py # training file for multi-graph datasets (PPI)
```
### Running the Code

These commands can be used to run the scripts with the main configurations reported in the paper:

```bash
python run_inductive.py \
        --project  gssl_lrgi --method lrgi --experiment products_main  \
        --dataset products --dataset_dir /path/to/products/dataset/ \
        --num_layers 3 --emb_size 128 --heads 4 \
        --lambda_1 25 --lambda_2 10 --lambda_3 1 \
        --K 1 --hid 8 \
        --p_drop_x 0.0 --p_drop_u 0.0  \
        --lr 1e-4 --epochs 100 --wd 1e-5 --batch_size 512
```

```bash
python run_inductive.py \
        --project  gssl_lrgi --method lrgi --experiment reddit_main  \
        --dataset reddit --dataset_dir /path/to/reddit/dataset/ \
        --num_layers 2 --emb_size 512 --heads 4 \
        --lambda_1 50 --lambda_2 25 --lambda_3 10 \
        --K 1 --hid 8 \
        --p_drop_x 0.0 --p_drop_u 0.0  \
        --lr 1e-4 --epochs 100 --wd 1e-5 --batch_size 512
```

```bash
python run_multigraph.py \
    --project gssl_lrgi --method lrgi --experiment ppi_main \
    --dataset_dir /path/to/ppi/dataset \
    --num_layers 3 --emb_size 1024 --heads 4 \
    --lambda_1 25 --lambda_2 25 --lambda_3 15 \
    --K 1 --hid 8 \
    --p_drop_x 0.0 --p_drop_u 0.0  \
    --lr 1e-4 --epochs 1000 --wd 1e-4 --batch_size 1
```



