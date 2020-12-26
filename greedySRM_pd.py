import torch
from tqdm import tqdm
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

# args = parser.parse_args()
parser = argparse.ArgumentParser(description='Greedy_SRM_old')
parser.add_argument('--runs',type=int, default=1)
parser.add_argument('--gnn', type=str, default='GCN')
parser.add_argument('--reset',type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--dataset',type=str, default='protein')
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--iter', type=int, default=10)
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--early', type=int, default=80)

args = parser.parse_args()
num_gnns = args.num_gnns
gnn = args.gnn
data_n = args.dataset
layer = args.layers
metrics = 'f1' if data_n == 'protein' else 'acc'

if data_n == 'protein':
    dataset = PygNodePropPredDataset('ogbn-proteins', root='../data')
elif data_n == 'product':
    dataset = PygNodePropPredDataset('ogbn-products', root='../data/products')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
#data.node_species = None
data.y = data.y.to(torch.float)
log_name = f'AdaGNN_{gnn}_dataset_{dataset_n}_weak_layers_{weak_layer}_'
# log_name = f'logs_version{version}_{times}'
# Initialize features of nodes by aggregating edge features.
row, col = data.edge_index
if data_n == 'protein':
    data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='add')
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


train_loader = RandomNodeSampler(data, num_parts=20, shuffle=True,
                                 num_workers=5)
test_loader = RandomNodeSampler(data, num_parts=5, num_workers=5)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AdaGNN_h(in_channels=data.x.size(-1), hidden_channels=64, num_layer_list=[layer] * 20, out_channels=data.y.size(-1), gnn_model=[gnn] * 20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = torch.nn.BCEWithLogitsLoss()
evaluator = Evaluator(data_n)

logger.add(log_name)
logger.info('logname: {}'.format(log_name))
criterion = nn.BCEWithLogitsLoss() if data_n == 'protein' else nn.CrossEntropyLoss()

def train(epoch, ith):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, k=ith)
        loss = criterion(out[data.train_mask], data.y[data.train_mask], reduction='none')
        loss = torch.sum(loss * weight[map_[data.n_id[data.train_mask]]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)*int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)
 
    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def evaluate(weight, metric):
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}
    
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, -1)
        m = data.train_mask

        if data_n == 'protein':

        pred_lb = out.argmax(dim=0,keepdim=True)
        outp = pred_lb
        m = data.train_mask
        err = (pred_lb[m][0]!=y[m]).float()
        err[err==0] = -1
        y_s = deepcopy(data.y)
        y_s[y_s == 0] = -1
        # weight[map_[data.n_id[data.train_mask]]] = 1 / (1 + torch.exp(y_s[data.train_mask] * out[data.train_mask]))
        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

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

    return train_m, valid_m, test_m

@torch.no_grad()
def test(ind,metric):
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    y_pred_t = torch.zeros(weight.size()).cuda()
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr,ind)

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        y_pred_t[map_[data.n_id[data.train_mask]]] = out[data.train_mask]

    y_pred_t[y_pred_t>0] = 1
    y_pred_t[y_pred_t<0] = 0
    err = (y_pred_t != y_tar)
    w_err = err.int() * weight
    w_err = w_err.sum()
    y_true_t = torch.cat(y_true['train'], dim=0)
    y_pred_t = torch.cat(y_pred['train'], dim=0)
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

# evaluator.num_tasks = data.y.size(1)
# evaluator.eval_metric = 'acc'
if data_n == 'protein':
    weight = torch.ones(train_cnt, data.y.size(-1)).cuda()
else:
    weight = torch.ones(train_cnt).cuda()
weight = weight / weight.sum()
train_list = []
val_list = []
test_list = []
num_list = []
epoch_list = []
gnn_cnt = []
for ind in range(num_gnns):
    best_train = 0
    best_w_err = 1
    best_val_epoch = 0
    w_err =0
    for epoch in range(1, 800):
        loss = train(epoch,ind)
        train_rocauc, valid_rocauc, test_rocauc, w_err = test(ind,metrics)
        if w_err < best_w_err:
            best_val = valid_rocauc
            best_w_err = w_err
            best_val_epoch = epoch
            logger.info(f'num_gnns: {ind+1}, epochs: {epoch},train_{metrics}: {train_rocauc:.4f} new best weighted error: {best_w_err:.4f}, test_{metrics}: {test_rocauc:.4f}')
            train_list.append(train_rocauc)
            val_list.append(valid_rocauc)
            test_list.append(test_rocauc)
            gnn_cnt.append(ind+1)
            epoch_list.append(epoch)
        print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
            f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}, Weighted_Error: {w_err:.4f}')
        if epoch - best_val_epoch > 30:
            break
    model.alpha[ind] = 0.5*torch.log2((1-w_err)/w_err)
    print(f'alpha_{ind} = {model.alpha}')
    train_rocauc, valid_rocauc, test_rocauc = evaluate(weight,metrics)
    weight = weight/weight.sum()
    logger.info(f'Evaluate : NumGNNs :{ind+1}, Train_{metrics}:{train_rocauc:.4f},Valid_{metrics}:{valid_rocauc:.4f}, Test_{metrics}:{test_rocauc:.4f}')

# df = pd.DataFrame({'layers':layer_list, 'epochs':epoch_list, 'train':train_list, 'val': val_list, 'test': test_list})
# df.to_pickle('vanilla_dgcn')