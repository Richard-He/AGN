import os.path as osp
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CitationFull, Reddit, Yelp, Flickr
import torch_geometric.transforms as T
# from torch_geometric.nn import SplineConv
from layer import AdaGNN_v, GAT, SAGE, GCN, MLP
from torch_geometric.nn import GENConv, DeepGCNLayer
# from torch_geometric.data import RandomNodeSampler
from loguru import logger
import numpy as np
# from utils import Pruner
import pickle
import argparse
parser = argparse.ArgumentParser(description='Greedy_SRM_old')
parser.add_argument('--runs',type=int, default=1)
parser.add_argument('--gnn', type=str, default='MLP')
parser.add_argument('--reset',type=lambda x: (str(x).lower() == 'true'), default=False)

parser.add_argument('--dataset',type=str, default='Cora')

parser.add_argument('--ratio', type=float, default=0.1)
parser.add_argument('--layers', type=int, default=30)
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--early', type=int, default=80)


args = parser.parse_args()
gnn = args.gnn
gnndict = {'GAT': GAT, 'SAGE': SAGE, 'GCN': GCN, 'GEN': AdaGNN_v, 'MLP': MLP}
reset = args.reset
ratio = args.ratio
dataset_n = args.dataset
t_layers = args.layers
log_name = f'./result/Greedy_SRM_GNN_{gnn}_reset_{reset}_dataset_{dataset_n}'
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset_n)
if dataset_n == 'dblp':
    dataset = CitationFull(path, dataset_n)
else:
    dataset = Planetoid(path, dataset_n)
data = dataset[0]
train_split = pickle.load(open(f'./datasetsplit/{dataset_n.lower()}_train', "rb") )
test_split = pickle.load(open(f'./datasetsplit/{dataset_n.lower()}_train', "rb") )
rand = torch.cat([train_split, test_split])
thold = int(data.num_nodes * ratio)
train_split = rand[:thold]
test_split = rand[thold:]
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[train_split] = 1
data.val_mask = None
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[test_split] = 1
criteria = CrossEntropyLoss()

out_channels = (torch.max(data.y) + 1).item()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = gnndict[gnn](in_channels=data.x.size(-1), hidden_channels=64, num_layers=t_layers, dropout=0.5, out_channels=out_channels).to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
logger.add(log_name)

def train():
    model.train()
    optimizer.zero_grad()
    criteria( model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask] ).backward()
    optimizer.step()

@torch.no_grad()
def test():
    model.eval()
    log_probs, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

num_layers = torch.arange(1,t_layers+1)
best_val = 0
train_list = []
val_list = []
test_list = []
layer_list = []
epoch_list = []
for layers in num_layers:
    r_list = []
    model.unfix(layers)
    for i in range(args.runs):
        if reset == True:
            model.reset_parameters()
            best_train = 0
            best_val = 0
            best_val_epoch = 0
        for epoch in range(1, args.epochs):
            train()
            train_acc, test_acc = test()
            if test_acc > best_val :
                best_val = test_acc
                best_val_epoch = epoch
                #logger.info(f'num_layers:{layers}, epochs: {epoch}, train: {train_acc:.4f}, new_best_val: {test_acc:.4f}')
            log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_acc, test_acc))
            if epoch - best_val_epoch > args.early:
                r_list.append(best_val)
                break
    logger.info(f'num_layers:{layers}, epochs: {best_val_epoch}, best_val: {np.mean(r_list):.4f}, std: {np.std(r_list):.4f}')
