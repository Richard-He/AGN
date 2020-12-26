import os.path as osp
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, CitationFull, Yelp, Flickr, PPI, Amazon
import torch_geometric.transforms as T
# from torch_geometric.nn import SplineConv
from layer import AdaGNN_v, GAT, SAGE, GCN
from torch_geometric.nn import GENConv, DeepGCNLayer
from ogb.nodeproppred import PygNodePropPredDataset
# from torch_geometric.data import RandomNodeSampler
from loguru import logger
import pickle


# dataset = 'cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data')
dataset = PygNodePropPredDataset('ogbn-products', root='../data/products')
data = dataset[0]
print(data.edge_index)
print(data.edge_attr==None)
print(data.x.size())
print(data.y.size())
print(data.y)
# dataset = Yelp(path + 'yelp/')
# data = dataset[0]
# print(data.edge_index)
# print(data.edge_attr==None)
# print(data.x.size())
# print(data.y.size())
# print(data.y)
# dataset = Flickr(path+ 'flickr/')
# data = dataset[0]
# print(data.edge_index)
# print(data.edge_attr==None)
# print(data.x.size())
# print(data.y.size())
# print(data.y)
# dataset = PygNodePropPredDataset('ogbn-products', root='../data')

# dataset = Amazon(path + 'amazon/', name='Computers')
# data = dataset[0]
# print(data.edge_index)
# print(data.edge_attr==None)
# print(data.x.size())
# print(data.y.size())
# print(data.y)
# thold = int(data.num_nodes / 2)
# train_mask = torch.randperm(data.num_nodes)[:thold]
# test_mask = torch.randperm(data.num_nodes)[thold:]
# pickle.dump(train_mask, open(f"./datasetsplit/cora_o_train", "wb") )
# pickle.dump(test_mask, open(f"./datasetsplit/cora_o_test", "wb") )