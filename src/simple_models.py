import torch
from torch import nn


class LSTMnet(nn.Module):
    def __init__(self, input_dim, output_dim, timsteps, cat_info=None, n_units=128, dropout=0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, n_units, batch_first=True)
        self.linear1 = nn.Linear(n_units, 2*n_units)
        self.linear2 = nn.Linear(2*n_units, output_dim)
        self.lrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)
        
        
    def forward(self, x):
        output, (h, c) = self.lstm(x)
        output = self.linear2(self.dropout(self.lrelu(self.linear1(output))))
        return torch.tanh(output)
    
    
class SLP(nn.Module):
    def __init__(self, input_dim, output_dim, timesteps, cat_info=None,):
        super().__init__()
        self.layer = nn.Linear(input_dim*timesteps, output_dim*timesteps)
        self.timesteps = timesteps
        self.output_dim = output_dim
        
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        out = self.layer(x)
        out = out.view(out.shape[0], self.timesteps, self.output_dim)
        out = torch.tanh(out)
        return out
    
    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, timesteps, cat_info=None, mult=0.3):
        super().__init__()
        
        hidden_dim = int(input_dim*timesteps*mult)
        
        self.hidden_layer = nn.Linear(input_dim*timesteps, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim*timesteps)
        self.timesteps = timesteps
        self.output_dim = output_dim
        
    
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.bn(self.hidden_layer(x))
        x = torch.tanh(x)
        out = self.output_layer(x)
        out = out.view(out.shape[0], self.timesteps, self.output_dim)
        out = torch.tanh(out)
        return out
    
    
class TCNTemporalBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, dilation):
        super().__init__()
        padding = int(dilation*(kernel_size-1))
        self.pad = nn.ConstantPad1d((padding, 0), 0)
        self.convbn1 = nn.Sequential(nn.Conv1d(input_channels, output_channels, kernel_size, dilation=dilation),
                                     nn.BatchNorm1d(output_channels))
        self.convbn2 = nn.Sequential(nn.Conv1d(output_channels, output_channels, kernel_size, dilation=dilation),
                                     nn.BatchNorm1d(output_channels))

        self.residual = nn.Conv1d(input_channels, output_channels, 1)
        self.lrelu = nn.LeakyReLU()
        
        
    def forward(self, x):
        out = self.pad(x)
        out = self.lrelu(self.convbn1(out))
        out = self.pad(out)
        out = self.lrelu(self.convbn2(out))
        y = self.residual(x)
        out = self.lrelu(out + y)
        return out
    

class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, timesteps, cat_info=None, n_layers=5, n_channels=64, kernel_size=3):
        super().__init__()
        self.first_layer = TCNTemporalBlock(input_dim, n_channels, kernel_size, 1)
        self.tcn_layers = nn.ModuleList([TCNTemporalBlock(n_channels, n_channels, kernel_size, 2**(i+1)) \
                                         for i in range(n_layers-1)])
        self.n_layers = n_layers
        self.output_transform = nn.Linear(n_channels, output_dim)
        
        
    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch_dim, lookback_window, n_features) -> (batch_dim, n_features, lookback_window)
        x = self.first_layer(x)
        for i in range(self.n_layers-1):
            x = self.tcn_layers[i](x)
        x = x.permute(0, 2, 1) # (batch_dim, n_features, lookback_window) -> (batch_dim, lookback_window, n_features)
        out = self.output_transform(x)
        out = torch.tanh(out)
        return out    