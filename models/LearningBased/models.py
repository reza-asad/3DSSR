import torch
import torch.nn as nn


class Lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, num_layers=2, num_direction=2):
        super(Lstm, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.num_layers = num_layers
        self.num_direction = num_direction

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=num_direction == 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, hidden):
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = self.dropout(lstm_out)
        return lstm_out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers * self.num_direction, batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.num_layers * self.num_direction, batch_size, self.hidden_dim).zero_().to(self.device))
        return hidden


class CosSinRegressor(nn.Module):
    def __init__(self, hidden_dim):
        super(CosSinRegressor, self).__init__()
        self.lin_layer = nn.Linear(hidden_dim * 2, 2, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        cos_sin = torch.tanh(self.lin_layer(x))

        return cos_sin


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
            embedding.unsqueeze_(dim=0)
            embeddings.append(embedding)
        H_in = torch.cat(embeddings, 0)

        # applyng diffusion
        H_out = torch.matmul(adj, H_in)

        # summing the node embeddings across all edge types
        H_out = torch.sum(H_out, dim=0)

        if self.bias is not None:
            H_out += self.bias

        return self.act(H_out)


class GCN_RES(nn.Module):
    def __init__(self, n_in, n_h, activation, num_edge_types, num_layers):
        super(GCN_RES, self).__init__()
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gcn_layers += [GCN(n_h, n_h, activation, num_edge_types)]
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
            H_1 = torch.cat([H_0, self.gcn_layers[i](H_0, adj)], dim=1)
            delta = self.lin_layer2(H_1)
            H_out = H_0 + delta

        return H_out


class Classifier(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.lin_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.lin_layer(x)

        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.lin_layer1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.lin_layer2 = nn.Linear(hidden_dim, output_dim, bias=False)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        x = torch.relu(self.lin_layer1(x))
        x = self.lin_layer2(x)

        return x
