import torch.nn as nn 
from torch.nn import LSTM
from collections import OrderedDict

class LSTM_(nn.Module):
    def __init__(self, input_shape, class_size, hidden_size, fc_sizes, droprate, 
                 num_layers=1, bidirectional=False):
        super(LSTM_, self).__init__() 
        
        model = []
        # input shape is bs x sl x dim
        
        # LSTM Layers 
        self.lstm_layer = LSTM(input_shape[2], hidden_size, num_layers=num_layers, 
                             dropout=droprate, bidirectional=bool(bidirectional)) 
        
        # FC Layers
        for i in range(len(fc_sizes)):
            if i == 0: 
                insize = hidden_size * num_layers
                if bidirectional: insize *= 2
                fc_layer = nn.Linear(insize,fc_sizes[i])
                model.append(('flatten', nn.Flatten()))
                model.append(('dropout_fc'+str(i+1), nn.Dropout(droprate)))
                model.append(('fc'+str(i+1), fc_layer))
                model.append(('relu_fc'+str(i+1), nn.ReLU()))
            else:
                fc_layer = nn.Linear(fc_sizes[i-1],fc_sizes[i])
                model.append(('dropout_fc'+str(i+1), nn.Dropout(droprate)))
                model.append(('fc'+str(i+1), fc_layer))
                model.append(('relu_fc'+str(i+1), nn.ReLU())) 
        
        # Last Layer
        last_layer = nn.Linear(fc_sizes[-1], class_size)
        model.append(('last', last_layer))
        if class_size > 1:
            model.append(('softmax', nn.Softmax(dim=1)))
            
        self.model = nn.Sequential(OrderedDict(model))
        
    def forward(self, input): 
        input_t = input.transpose(0,1) # transpose to sl, bs, dim   
        lstm_out = self.lstm_layer(input_t)[1][0].transpose(0,1) # to bs, num_layers * dim
        return self.model(lstm_out)
    
    def __repr__(): 
        return 'LSTM_'