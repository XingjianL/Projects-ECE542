# pytorch rnn tutorial
import torch.nn as nn
import torch
    
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, interval, batch_first = True, all_output = False):
        super(GRUModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.interval = interval
        self.all_output = all_output

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first = batch_first)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
        self.only_one = nn.Sequential(nn.Conv1d(output_size, output_size*2, kernel_size=15, padding="same"),
                                      nn.BatchNorm1d(output_size*2),
                                      nn.Conv1d(output_size*2, output_size, kernel_size=15, padding="same"),
                                      nn.Linear(interval,10),
                                      nn.Linear(10,1))

    def forward(self, x : torch.Tensor, h0 : torch.Tensor):
        if h0 is None:
            self.init_hidden(x.shape[0])
        else:
            self.h0 = h0
        out, h1 = self.gru(x, self.h0)
        out = self.fc(self.relu(out)).swapaxes(1,2)
        if self.all_output:
            return out, h1
        out = self.only_one(out)
        return out, h1
    def init_hidden(self, batch_size):
        self.h0 = torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda()