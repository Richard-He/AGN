import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GENConv, GATConv, SAGEConv

class AdaGNNLayer(torch.nn.Module):
    def __init__(self, conv=None, norm=None, act=None,
                    dropout=0., ckpt_grad=False):
        super(AdaGNNLayer, self).__init__()

        self.conv = conv
        self.norm = norm
        self.act = act
        self.dropout = dropout
        self.ckpt_grad = ckpt_grad
        dim = conv.out_channels
        # self.lam = torch.nn.Parameter(torch.zeros(1))
        self.fixed = True

    def unfix(self):
        self.fixed = False

    def forward(self, *args, **kwargs):
        args = list(args)
        x = args.pop(0)
        if self.fixed == True:
            return x
        else:
            if self.norm is not None:
                h = self.norm(x)
            if self.act is not None:
                h = self.act(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.conv is not None and self.ckpt_grad and h.requires_grad:
                h = checkpoint(self.conv, h, *args, **kwargs)
            else:
                h = self.conv(h, *args, **kwargs)

            return h + x

    def reset_parameters(self):
        self.conv.reset_parameters()

    def __repr__(self):
        return '{}(block={})'.format(self.__class__.__name__, self.block)

class SAGE(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()
        self.edge_encoder = Linear(in_channels, hidden_channels)
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.layers = torch.nn.ModuleList()

        for i in range(1, num_layers+1):
            conv = SAGEConv(hidden_channels, hidden_channels)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = AdaGNNLayer(conv,norm,act,dropout=0.1)
            self.layers.append(layer)
        
        self.lin = Linear(hidden_channels, out_channels)
        self.currenlayer = 1
        self.layers[0].unfix()
        self.num_layers = num_layers
    
    def get_fixed_layer_cnt(self):
        return self.currenlayer

    def unfix(self, until_num_layer=1):
        for i in range(until_num_layer-self.currenlayer):
            if self.currenlayer >= len(self.layers):
                return
            else:
                self.layers[self.currenlayer].unfix()
                self.currenlayer +=1

    
    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        # edge_attr = self.edge_encoder(edge_attr)
        #print(x.size(), edge_index.size())
        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.layers[0].norm(x)
        x = self.layers[0].act(x)
        x = F.dropout(x, p=0.1, training=self.training)
        return self.lin(x)
    

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin.reset_parameters()
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()

class GAT(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels, num_layers):
        super(GAT, self).__init__()
        self.edge_encoder = Linear(in_channels, hidden_channels)
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.layers = torch.nn.ModuleList()

        for i in range(1, num_layers+1):
            conv = GATConv(hidden_channels, hidden_channels)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = AdaGNNLayer(conv,norm,act,dropout=0.1)
            self.layers.append(layer)
        
        self.lin = Linear(hidden_channels, out_channels)
        self.currenlayer = 1
        self.layers[0].unfix()
        self.num_layers = num_layers
    
    def get_fixed_layer_cnt(self):
        return self.currenlayer

    def unfix(self, until_num_layer=1):
        for i in range(until_num_layer-self.currenlayer):
            if self.currenlayer >= len(self.layers):
                return
            else:
                self.layers[self.currenlayer].unfix()
                self.currenlayer +=1

    
    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        # edge_attr = self.edge_encoder(edge_attr)
        #print(x.size(), edge_index.size())
        x = self.layers[0].conv(x, edge_index)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.layers[0].norm(x)
        x = self.layers[0].act(x)
        x = F.dropout(x, p=0.1, training=self.training)
        return self.lin(x)
    

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin.reset_parameters()
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()


class WeakGNN(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels, num_layers):
        super(WeakGNN,self).__init__()
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.edge_encoder = Linear(in_channels, hidden_channels)
        self.layers=torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                            t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = AdaGNNLayer(conv, norm, act, dropout=0.1, ckpt_grad=False)
            self.layers.append(layer)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.node_encoder(x)
        if edge_attr != None:
            edge_attr = self.edge_encoder(edge_attr)
        x = self.layers[0].conv(x, edge_index, edge_attr)
        if len(self.layers) > 1:
            for layer in self.layers[1:]:
                x = layer(x, edge_index, edge_attr)

        x = self.layers[0].norm(x)
        x = self.layers[0].act(x)
        x = F.dropout(x, p=0.1, training=self.training)
        return self.lin(x)
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin.reset_parameters()
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()

class AdaGNN_h(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_gnns, num_layer_list):
        super(AdaGNN_h, self).__init__()

        self.gnns = torch.nn.ModuleList()
        assert len(num_layer_list) == num_gnns
        self.num_layer_list = num_layer_list
        for i in range(num_gnns):
            gnn = WeakGNN(in_channels, hidden_channels, out_channels,num_layer_list[i])
            self.gnns.append(gnn)
        
        self.alpha = torch.zeros(num_gnns)
        self.alpha[0] = 1


    def forward(self,x, edge_index, edge_attr, k=0):
        if k == -1:
            with torch.no_grad():
                s = self.alpha[0] * self.gnns[0](x,edge_index,edge_attr)
                for i in range(len(self.gnns)):
                    if self.alpha[i] == 0:
                        break
                    else:
                        s += self.alpha[i]*self.gnns[i](x,edge_index,edge_attr)
                return s
        elif k < len(self.gnns):
            s = self.gnns[k](x, edge_index, edge_attr)
            return s
    
    def reset_parameters(self):
        for gnn in self.gnns:
            gnn.reset_parameters()
        self.alpha = 0



class AdaGNN_v(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels, num_layers):
        super(AdaGNN_v, self).__init__()

        self.node_encoder = Linear(in_channels, hidden_channels)
        self.edge_encoder = Linear(in_channels, hidden_channels)

        self.layers = torch.nn.ModuleList()

        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                            t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = AdaGNNLayer(conv, norm, act, dropout=0.1, ckpt_grad=False)
            self.layers.append(layer)
        
        self.lin = Linear(hidden_channels, out_channels)
        self.currenlayer = 1
        self.layers[0].unfix()
        self.num_layers = num_layers
        

    def get_fixed_layer_cnt(self):

        return self.currenlayer

    def unfix(self, until_num_layer=1):
        for i in range(until_num_layer-self.currenlayer):
            if self.currenlayer >= len(self.layers):
                return
            else:
                self.layers[self.currenlayer].unfix()
                self.currenlayer +=1

    
    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        x = self.layers[0].conv(x, edge_index, edge_attr)
        if len(self.layers) > 1:
            for layer in self.layers[1:]:
                x = layer(x, edge_index, edge_attr)

        x = self.layers[0].norm(x)
        x = self.layers[0].act(x)
        x = F.dropout(x, p=0.1, training=self.training)
        return self.lin(x)
    

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.lin.reset_parameters()
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()