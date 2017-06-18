import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hidden, cell):
        shape = input.size()
        seq_len = shape[0]
        batch_size = shape[1]

        output, h_c = self.lstm(input, (hidden, cell))
        output = output.view(seq_len * batch_size, self.hidden_size)
        output = self.decoder(output)
        output = output.view(seq_len, batch_size, self.output_size)
        return output, h_c

    def init_hidden_cell(self):
        hidden =  Variable(torch.zeros(self.n_layers, 1, self.hidden_size).cuda())
        cell = Variable(torch.zeros(self.n_layers, 1, self.hidden_size).cuda())
        return (hidden, cell)

