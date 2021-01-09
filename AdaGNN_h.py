import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from ogb.nodeproppred import PygNodePropPredDataset
from evaluate import Evaluator
from layer import AdaGNN_h
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.data import RandomNodeSampler
from torch_geometric.datasets import Reddit, Yelp, Flickr
from loguru import logger
from copy import deepcopy
import argparse
import pandas as pd

# args = parser.parse_args()
parser = argparse.ArgumentParser(description='AdaGNN')
parser.add_argument('--gnn', type=str, default='GCN')
parser.add_argument('--dataset',type=str, default='protein')
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--early', type=int, default=30)
parser.add_argument('--num_gnns', type=int, default=20)
parser.add_argument('--num_train_parts', type=int, default=40)
parser.add_argument('--num_test_parts', type=int, default=10)

args = parser.parse_args()
num_gnns = args.num_gnns
gnn = args.gnn
data_n = args.dataset
layer = args.layers
epochs = args.epochs
metrics = 'f1' if data_n == 'protein' else 'acc'

if data_n == 'protein':
    dataset = PygNodePropPredDataset('ogbn-proteins', root='../data')
elif data_n == 'product':
    dataset = PygNodePropPredDataset('ogbn-products', root='../data/products')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.n_id = torch.arange(data.num_nodes)
#data.node_species = None
if data_n == 'protein':
    row, col = data.edge_index
    data.y = data.y.to(torch.float)
    data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='add')
elif data_n == 'product':
    data.y = data.y.squeeze()
log_name = f'./result/AdaGNN_{gnn}_dataset_{data_n}_weak_layers_{layer}_{args.num_train_parts}loss'
# log_name = f'logs_version{version}_{times}'
# Initialize features of nodes by aggregating edge features.
    
# Set split indices to masks.
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask
data['test_mask'] = data['valid_mask'] | data['test_mask']
y_tar = data.y[data.train_mask].cuda()

map_ = torch.zeros(data.num_nodes, dtype=torch.long)
train_cnt = data['train_mask'].int().sum()
map_[splitted_idx['train']] = torch.arange(train_cnt)


train_loader = RandomNodeSampler(data, num_parts=args.num_train_parts, shuffle=True,
                                 num_workers=5)
test_loader = RandomNodeSampler(data, num_parts=args.num_test_parts, num_workers=5)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if data_n == 'protein':
    model = AdaGNN_h(in_channels=data.x.size(-1), hidden_channels=64, num_layer_list=[layer] * num_gnns, out_channels=data.y.size(-1), gnn_model=[gnn] * num_gnns).to(device)
elif data_n == 'product':
    model = AdaGNN_h(in_channels=data.x.size(-1), hidden_channels=64, num_layer_list=[layer] * num_gnns, out_channels=data.y.max().int().item()+1, gnn_model=[gnn] * num_gnns).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.BCEWithLogitsLoss()
if data_n == 'product':
    evaluator = Evaluator(data_n,t_ype=2)
else:
    evaluator = Evaluator(data_n)

logger.add(log_name)
logger.info('logname: {}'.format(log_name))
criterion = nn.BCEWithLogitsLoss(reduction='none') if data_n == 'protein' else nn.CrossEntropyLoss(reduction='none')
if data_n == 'protein':
    weight = torch.ones(train_cnt, data.y.size(-1)).cuda()
elif data_n == 'product':
    weight = torch.ones(train_cnt).cuda()
def train(epoch, ith):
    model.train()

    # pbar = tqdm(total=len(train_loader))
    # pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, k=ith)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss = torch.sum(loss * weight[map_[data.n_id[data.train_mask]]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)*int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        # pbar.update(1)
 
    # pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def evaluate(metric):
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}
    
    total_loss_train = total_loss_test = total_examples_train = total_examples_test = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, -1)
        total_loss_train += float(criterion(out[data.train_mask], data.y[data.train_mask]).mean())*int(data.train_mask.sum())
        total_examples_train += int(data.train_mask.sum())
        total_loss_test += float(criterion(out[data.test_mask], data.y[data.test_mask]).mean())*int(data.test_mask.sum())
        total_examples_test += int(data.test_mask.sum())
        m = data.train_mask
        if data_n == 'protein':
            y_s = deepcopy(data.y)
            y_s[y_s == 0] = -1
            #weight[map_[data.n_id[m]]] = weight[map_[data.n_id[m]]] * torch.exp(-y_s[m] * 2*(torch.sigmoid(out[m])-0.5))
            weight[map_[data.n_id[data.train_mask]]] = 1 / (1 + torch.exp(y_s[m] * out[m]))

            for split in y_true.keys():
                mask = data[f'{split}_mask']
                y_true[split].append(data.y[mask].cpu())
                y_pred[split].append(out[mask].cpu())
        
        elif data_n == 'product':
            out = F.softmax(out)
            vals, pred_lb = torch.max(out,1)
            right = (pred_lb[m] == data.y[m].squeeze()).int()
            right[right==0] = -1
            # ind = torch.stack([torch.arange(m.int().sum()).cuda(), (data.y)[m].int().squeeze()]).T            
            pred_val = torch.gather(out[m],1, (data.y)[m].long().unsqueeze(-1)).squeeze()
            # weight[map_[data.n_id[data.train_mask]]] = weight[map_[data.n_id[data.train_mask]]] * torch.exp(-right* pred_val)
            weight[map_[data.n_id[data.train_mask]]] = 1 / (1 + torch.exp(right* pred_val))
            for split in y_true.keys():
                mask = data[f'{split}_mask']
                y_true[split].append(data.y[mask].cpu())
                y_pred[split].append(pred_lb[mask].cpu())

    train_m = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    }, metric)

    valid_m = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    }, metric)

    test_m = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    }, metric)

    return train_m, valid_m, test_m, total_loss_train / total_examples_train, total_loss_test / total_examples_test
    

@torch.no_grad()
def test(ind,metric):
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}
    if data_n == 'protein':
        y_pred_t = torch.zeros(weight.size(),dtype=torch.float).cuda()
    elif data_n == 'product':
        y_pred_t = torch.zeros(weight.size(),dtype=torch.long).cuda()
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr,ind)
        
        if data_n == 'protein':
            for split in y_true.keys():
                mask = data[f'{split}_mask']
                y_true[split].append(data.y[mask].cpu())
                y_pred[split].append(out[mask].cpu())
            y_pred_t[map_[data.n_id[data.train_mask]]] = out[data.train_mask]
            

        elif data_n == 'product':
            vals, pred_lb = torch.max(out, 1)
            for split in y_true.keys():
                mask = data[f'{split}_mask']
                y_true[split].append(data.y[mask].cpu())
                y_pred[split].append(pred_lb[mask].cpu())
            y_pred_t[map_[data.n_id[data.train_mask]]] = pred_lb[data.train_mask]
            
    if data_n == 'protein':
        y_pred_t[y_pred_t>0] = 1
        y_pred_t[y_pred_t<0] = 0
        err = (y_pred_t != y_tar)
        w_err = err.int() * weight
        w_err = w_err.sum()
    elif data_n == 'product':
        err = (y_pred_t != y_tar)
        w_err = err.int() * weight
        w_err = w_err.sum()

    train_m = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    }, metric)

    valid_m = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    }, metric)

    test_m = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    }, metric)

    return train_m, valid_m, test_m, w_err
    # y_true_t = torch.cat(y_true['train'], dim=0)
    # y_pred_t = torch.cat(y_pred['train'], dim=0)
    

# evaluator.num_tasks = data.y.size(1)
# evaluator.eval_metric = 'acc'


weight = weight / weight.sum()
train_list = []
val_list = []
test_list = []
num_list = []
epoch_list = []
gnn_cnt = []
tr_losses = []
te_losses = []
ep = []
layr = []
for ind in range(num_gnns):
    best_train = 0
    best_w_err = 1
    best_val_epoch = 0
    w_err =0
    for epoch in range(1, epochs+1):
        loss = train(epoch,ind)
        train_rocauc, valid_rocauc, test_rocauc, w_err = test(ind,metrics)
        if w_err < best_w_err:
            best_val = valid_rocauc
            best_w_err = w_err
            best_val_epoch = epoch
            #logger.info(f'num_gnns: {ind+1}, epochs: {epoch},train_{metrics}: {train_rocauc:.4f} new best weighted error: {best_w_err:.4f}, test_{metrics}: {test_rocauc:.4f}')
            train_list.append(train_rocauc)
            val_list.append(valid_rocauc)
            test_list.append(test_rocauc)
            gnn_cnt.append(ind+1)
            epoch_list.append(epoch)
        print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
            f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}, Weighted_Error: {w_err:.4f}')
        if epoch - best_val_epoch > args.early or epoch==epochs :
            ep.append(epoch)
            layr.append(ind)
            break
    model.alpha[ind] = 0.5*torch.log2((1-w_err)/w_err)
    #print(f'alpha_{ind} = {model.alpha}')
    train_rocauc, valid_rocauc, test_rocauc, loss_tr, loss_te = evaluate(metrics)
    tr_losses.append(loss_tr)
    te_losses.append(loss_te)
    weight = weight/weight.sum()
    logger.info(f' NumGNNs :{ind+1}, Train_{metrics}, train_loss:{loss_tr:.4f}, test_loss{loss_te:.4f}')
    #logger.info(f'Evaluate : NumGNNs :{ind+1}, Train_{metrics}:{train_rocauc:.4f},Valid_{metrics}:{valid_rocauc:.4f}, Test_{metrics}:{test_rocauc:.4f}')
t_dic = {'num_gnns': layr, 'epochs': ep, 'train_loss':tr_losses, 'test_loss':te_losses}
df = pd.DataFrame.from_dict(t_dic)
df.to_pickle(log_name+'_save_')
# df = pd.DataFrame({'layers':layer_list, 'epochs':epoch_list, 'train':train_list, 'val': val_list, 'test': test_list})
# df.to_pickle('vanilla_dgcn')