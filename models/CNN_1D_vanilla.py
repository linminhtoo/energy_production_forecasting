import torch 
import torch.nn as nn 
from collections import OrderedDict

class CNN_1D(nn.Module): 
    def __init__(self, input_shape, class_size, hidden_sizes, kernel_sizes, maxpool_size,
                fc_sizes, droprate):
        """
        input_shape(list): (batchsize, sequence length, feature dimensions)
        hidden_sizes(list): sizes of every hidden conv layer 
        kernel_sizes(list): sizes of kernel at every conv layer
        maxpool_size(int): sizes of maxpool (constant across different layers)
        fc_sizes(list): sizes of every fully connected layer after convolutions
        """
        
        super(CNN_1D, self).__init__()
        
        assert len(hidden_sizes) == len(kernel_sizes)
        
        model = []
        seqlen = input_shape[1]
        
        # Calculate length-preserving padding sizes 
        padding_sizes = []
        for k in kernel_sizes: 
            if k % 2 == 0: 
                padding_sizes.append((k // 2 - 1, k // 2))
            else: 
                padding_sizes.append(k // 2)
        
        # Conv Layers
        for i in range(len(hidden_sizes)): 
            if i == 0: 
                conv_layer = nn.Conv1d(input_shape[2], hidden_sizes[i], 
                                       kernel_sizes[i], stride=1,
                                       padding=padding_sizes[i])
                maxpool_layer = nn.MaxPool1d(maxpool_size,maxpool_size)
                model.append(('conv'+str(i+1), conv_layer))
                model.append(('relu_conv'+str(i+1), nn.ReLU()))
                model.append(('maxpool'+str(i+1), maxpool_layer))
            else:
                conv_layer = nn.Conv1d(hidden_sizes[i-1], hidden_sizes[i], 
                                       kernel_sizes[i], stride=1,
                                       padding=padding_sizes[i])
                maxpool_layer = nn.MaxPool1d(maxpool_size,maxpool_size)
                model.append(('conv'+str(i+1), conv_layer))
                model.append(('relu_conv'+str(i+1), nn.ReLU()))
                model.append(('maxpool'+str(i+1), maxpool_layer))
            seqlen = seqlen // maxpool_size
        
        # FC Layers
        for i in range(len(fc_sizes)):
            if i == 0: 
                insize = seqlen * hidden_sizes[-1]
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
        input_t = input.transpose(1,2) # transpose to bs, dim, sl  
        return self.model(input_t)
    
    def __repr__(): 
        return 'CNN_1D'
    