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
        self.only_one = nn.Sequential(nn.Conv1d(interval, int(interval/2), kernel_size=1, padding="same"),
                                      nn.BatchNorm1d(int(interval/2)),
                                      nn.Conv1d(int(interval/2), int(interval/4), kernel_size=1, padding="same"),
                                      nn.Conv1d(int(interval/4), 1, kernel_size=1, padding="same"),
                                      #nn.Linear(int(interval/4),10),
                                      #nn.Linear(10,1))
        )

    def forward(self, x : torch.Tensor, h0 : torch.Tensor):
        if h0 is None:
            self.init_hidden(x.shape[0])
        else:
            self.h0 = h0
        out, h1 = self.gru(x, self.h0)
        out = self.fc(self.relu(out))
        if self.all_output:
            return out.swapaxes(1,2), h1
        #out = self.only_one(out.swapaxes(1,2))
        #print(out.shape)
        out = self.only_one(out).swapaxes(1,2)
        #print(out.shape)
        return out, h1
        return out[:,-1,None].swapaxes(1,2), h1
    def init_hidden(self, batch_size):
        self.h0 = torch.zeros((self.num_layers, batch_size, self.hidden_size)).cuda()