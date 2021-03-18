import torch
import torch.nn as nn


class DeepSetAlign(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepSetAlign, self).__init__()
        self.lin_layer1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.lin_layer2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_layer3 = nn.Linear(hidden_dim, 2, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, features):
        hidden = torch.relu(self.lin_layer1(features))
        # hidden = hidden + torch.sin(hidden)**2
        hidden = torch.relu(self.lin_layer2(hidden))
        # hidden = hidden + torch.sin(hidden)**2
        summary = torch.mean(hidden, dim=1)

        # use the summary to predict cosine and sine of the best alignment angle
        cos_sin = torch.tanh(self.lin_layer3(summary))

        return cos_sin


# class LstmAlign(nn.Module):
#     def __init__(self, input_dim, hidden_dim, device):
#         super(LstmAlign, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.device = device
#
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
#
#     def forward(self, x, hidden):
#         lstm_out, hidden = self.lstm(x, hidden)
#         return lstm_out, hidden
#
#     def init_hidden(self, batch_size):
#         weight = next(self.parameters()).data
#         hidden = (weight.new(1, batch_size, self.hidden_dim).zero_().to(self.device),
#                   weight.new(1, batch_size, self.hidden_dim).zero_().to(self.device))
#         return hidden

class LstmAlign(nn.Module):
    def __init__(self, input_dim, hidden_dim, device, num_layers=2, num_direction=2):
        super(LstmAlign, self).__init__()
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


class LinLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(LinLayer, self).__init__()
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
