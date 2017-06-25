import torch
import torch.nn as nn
from torch.autograd import Variable

import pdb

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, batch_first=True)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input):
        shape = input.size()
        batch_size = shape[0]
        seq_len = shape[1]

        hidden =  Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())
        cell = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())

        output, h_c = self.lstm(input, (hidden, cell))

        output = output.contiguous().view(batch_size * seq_len, self.hidden_size)
        output = self.decoder(output)
        output = output.view(batch_size, seq_len, self.output_size)
        return output, h_c

