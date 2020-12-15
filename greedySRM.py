import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch_scatter import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from layer import AdaGNN_v, GAT, SAGE
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.data import RandomNodeSampler
from loguru import logger

# args = parser.parse_args()
log_name = 'Greedy_SRM_SAGE'
num_layers=32
dataset = PygNodePropPredDataset('ogbn-proteins', root='../data')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.node_species = None
data.y = data.y.to(torch.float)
reset=False
# log_name = f'logs_version{version}_{times}'
# Initialize features of nodes by aggregating edge features.
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, 0, dim_size=data.num_nodes, reduce='add')

# Set split indices to masks.
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask
data['test_mask'] = data['valid_mask'] | data['test_mask']

train_loader = RandomNodeSampler(data, num_parts=20, shuffle=True,
                                 num_workers=5)
test_loader = RandomNodeSampler(data, num_parts=5, num_workers=5)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(in_channels=data.x.size(-1), hidden_channels=64, num_layers=num_layers, out_channels=data.y.size(-1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
evaluator = Evaluator('ogbn-proteins')

logger.add(log_name)
logger.info('logname: {}'.format(log_name))

def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)
 
    pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test():
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)
        # out[out>0] = 1
        # out[out<0] = 0
        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())


    train_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['rocauc']

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['rocauc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc

# evaluator.num_tasks = data.y.size(1)
# evaluator.eval_metric = 'acc'
num_layers = torch.arange(1, 30, 2)
train_list = []
val_list = []
test_list = []
layer_list = []
epoch_list = []
for layers in num_layers:
    best_train = 0
    best_val = 0
    best_val_epoch = 0
    model.unfix(layers)
    if reset == True:
        model.reset_parameters()
    for epoch in range(1, 800):
        loss = train(epoch)
        train_rocauc, valid_rocauc, test_rocauc = test()
        if valid_rocauc > best_val:
            best_val = valid_rocauc
            best_val_epoch = epoch
            logger.info(f'num_layers: {layers}, epochs: {epoch},train: {train_rocauc:.4f} new best valid: {best_val:.4f}, test: {test_rocauc:.4f}')
            train_list.append(train_rocauc)
            val_list.append(valid_rocauc)
            test_list.append(test_rocauc)
            layer_list.append(layers)
            epoch_list.append(epoch)
        print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
            f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
        if epoch - best_val_epoch > 50:
            break
df = pd.DataFrame({'layers':layer_list, 'epochs':epoch_list, 'train':train_list, 'val': val_list, 'test': test_list})
df.to_pickle('vanilla_dgcn')