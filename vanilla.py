import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, ReLU
from torch_scatter import scatter
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.data import RandomNodeSampler
from loguru import logger

log_name = 'vanilla_deep_gcn'
logger.add(log_name)
# parser.add_argument('--lr', type=float, default=0.01)

# parser.add_argument('--model_n', type=str, default='deepgcn')
# parser.add_argument('--method', type=str, default='ada')
# parser.add_argument('--reset',type=lambda x: (str(x).lower() == 'true'), default=False)

# parser.add_argument('--num_test_parts',type=int, default=5)
# parser.add_argument('--num_parts',type=int, default=30)
# parser.add_argument('--times',type=int, default=20)

# parser.add_argument('--prune_epochs', type=int, default=250)
# parser.add_argument('--start_epochs', type=int, default=200)

# parser.add_argument('--num_workers', type=int, default=5)
# parser.add_argument('--ratio', type=float, default=0.95)
# parser.add_argument('--prune_set',type=str, default='train')
# parser.add_argument('--dropout',type=float, default=0.5)
# parser.add_argument('--data_dir',type=str,default='./data/')
# parser.add_argument('--savept',type=lambda x: (str(x).lower() == 'true'), default=False)
# parser.add_argument('--globe',type=lambda x: (str(x).lower() == 'true'), default=False)
# args = parser.parse_args()


dataset = PygNodePropPredDataset('ogbn-proteins', root='../data')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.node_species = None
data.y = data.y.to(torch.float)

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


class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super(DeeperGCN, self).__init__()

        self.node_encoder = Linear(data.x.size(-1), hidden_channels)
        self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, data.y.size(-1))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin.reset_parameters()
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].norm(x)
        x = self.layers[0].act(x)
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.BCEWithLogitsLoss()
evaluator = Evaluator('ogbn-proteins')


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

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')

    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr)

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

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
    model = DeeperGCN(hidden_channels=64, num_layers=layers).to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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
        if torch.abs(epoch - best_val_epoch) > 60:
            break
    del(model)
df = pd.DataFrame({'layers':layer_list, 'epochs':epoch_list, 'train':train_list, 'val': val_list, 'test': test_list})
df.to_pickle('vanilla_dgcn')