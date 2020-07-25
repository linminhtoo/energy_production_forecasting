import torch 
import torch.nn as nn
from collections import OrderedDict

class MLP(nn.Module): 
    def __init__(self, input_shape, class_size, fc_sizes, droprate):
        """
        input_shape(list): (batchsize, sequence length, feature dimensions)
        hidden_sizes(list): sizes of every hidden conv layer 
        kernel_sizes(list): sizes of kernel at every conv layer
        maxpool_size(int): sizes of maxpool (constant across different layers)
        fc_sizes(list): sizes of every fully connected layer after convolutions
        """
        
        super(MLP, self).__init__()
        
        model = []
        # FC Layers
        model.append(('flatten', nn.Flatten()))
        for i in range(len(fc_sizes)):
            if i == 0: 
                insize = input_shape[2] * input_shape[1]
                fc_layer = nn.Linear(insize ,fc_sizes[i])
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
        input_t = input.transpose(1,2) # transpose to bs, dim, sl  
        return self.model(input_t)
    
    def __repr__(): 
        return 'MLP'
    