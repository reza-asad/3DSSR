import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, num_edge_types, bias=True):
        super(GCN, self).__init__()
        # define the linear layers and the activation
        self.num_edge_types = num_edge_types
        self.lin_layers = nn.ModuleList()
        for i in range(num_edge_types):
            self.lin_layers.append(nn.Linear(in_ft, out_ft, bias=False))
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, x, adj):
        # deriving initial embedding
        embeddings = []
        for i in range(self.num_edge_types):
            embedding = self.lin_layers[i](x)
            embedding.unsqueeze_(dim=1)
            embeddings.append(embedding)
        H_in = torch.cat(embeddings, 1)

        # applyng diffusion
        H_out = torch.matmul(adj, H_in)

        # summing the node embeddings across all edge types
        H_out = torch.sum(H_out, dim=1)

        if self.bias is not None:
            H_out += self.bias

        return self.act(H_out)


class GCN_RES(nn.Module):
    def __init__(self, n_in, n_h, activation, num_edge_types, num_layers):
        super(GCN_RES, self).__init__()
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gcn_layers += [GCN(n_h, n_h, activation, num_edge_types, bias=False)]
        self.lin_layer1 = nn.Linear(n_in, n_h, bias=False)
        self.lin_layer2 = nn.Linear(2*n_h, n_h, bias=False)

        self.weights_init(self.lin_layer1)
        self.weights_init(self.lin_layer2)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, adj):
        H_0 = self.lin_layer1(x)
        for i in range(len(self.gcn_layers)):
            H_1 = torch.cat([H_0, self.gcn_layers[i](H_0, adj)], dim=2)
            delta = self.lin_layer2(H_1)
            H_out = H_0 + delta

        return H_out

    # Detach the return variables
    def embed(self, seq, adj):
        h_1 = self.fc1(seq)
        for i in range(len(self.gcn_layers)):
            h_1_2 = torch.cat([h_1, self.gcn_layers[i](h_1, adj)], dim=2)
            delta = self.fc2(h_1_2)
            h_1 = h_1 + delta
        return h_1.detach()


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(hidden_dim, hidden_dim, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, summary, hidden):
        summary = torch.unsqueeze(summary, 1)
        summary = summary.expand_as(hidden)
        logits = torch.squeeze(self.f_k(hidden, summary), 2)

        return logits


class LinearLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearLayer, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


