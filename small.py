import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv
from utils import Pruner

dataset = 'Cora'
style = 'truncate'

path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.TargetIndegree())
data = dataset[0]

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[:data.num_nodes - 1000] = 1
data.val_mask = None
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[data.num_nodes - 500:] = 1
pruner = Pruner(style=style, ratio1=0.8, ratio2=0.5)

class Net(torch.nn.Module):
    def __init__(self,prune=False):
        super(Net, self).__init__()
        self.conv1 = SplineConv(dataset.num_features, 16, dim=1, kernel_size=2)
        self.conv2 = SplineConv(16, dataset.num_classes, dim=1, kernel_size=2)
        self.prune=prune
        if prune=='data':
            self.prunedata()

    def prunedata(self):
        e_id = pruner.prune(data.edge_index)
        data.edge_index = data.edge_index[:,e_id]
        data.edge_attr = data.edge_attr[e_id]

    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        if self.prune ==True and self.training==True:
            e_id = pruner.prune(edge_index)
            edge_index = edge_index[:,e_id]
            edge_attr = edge_attr[e_id]
        x = F.dropout(x,p=0, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x,p=0, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net(prune='data').to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    log_probs, accs = model(), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

best_test=0
for epoch in range(1, 501):
    train()
    train_acc, test_acc = test() 
    log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, test_acc))
    if test_acc>best_test:
        print(f'best acc : {best_test:.4f}')
        best_test = test_acc