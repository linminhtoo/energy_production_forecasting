import torch.nn as nn 
from torch.nn import LSTM
from collections import OrderedDict


class CNN_LSTM_stacked(nn.Module): 
    def __init__(self, input_shape, class_size,
                 cnn_hidden_sizes, kernel_sizes, maxpool_size,
                 lstm_hidden_dim, num_layers, bidirectional,
                 fc_sizes, droprate):
        """
        input_shape(list): (batchsize, sequence length, feature dimensions)
        
        CNN
        cnn_hidden_sizes(list): sizes of every hidden conv layer 
        kernel_sizes(list): sizes of kernel at every conv layer
        maxpool_size(int): sizes of maxpool (constant across different layers)
        
        LSTM
        lstm_hidden_dim(int): number of hidden layers
        num_layers(int): number of lstm layers
        bidirectional(bool)
        
        FC
        fc_sizes(list): sizes of every fully connected layer after convolutions
        """
        
        super(CNN_LSTM_stacked, self).__init__()
        
        assert len(cnn_hidden_sizes) == len(kernel_sizes)
        
        cnn_model = []
        seqlen = input_shape[1]
        
        # Calculate length-preserving padding sizes 
        padding_sizes = []
        for k in kernel_sizes: 
            if k % 2 == 0: 
                padding_sizes.append((k // 2 - 1, k // 2))
            else: 
                padding_sizes.append(k // 2)
        
        # Conv Layers
        for i in range(len(cnn_hidden_sizes)): 
            if i == 0: 
                conv_layer = nn.Conv1d(input_shape[2], cnn_hidden_sizes[i], 
                                       kernel_sizes[i], stride=1,
                                       padding=padding_sizes[i])
                maxpool_layer = nn.MaxPool1d(maxpool_size,maxpool_size)
                cnn_model.append(('conv'+str(i+1), conv_layer))
                cnn_model.append(('relu_conv'+str(i+1), nn.ReLU()))
                cnn_model.append(('maxpool'+str(i+1), maxpool_layer))
            else:
                conv_layer = nn.Conv1d(cnn_hidden_sizes[i-1], cnn_hidden_sizes[i], 
                                       kernel_sizes[i], stride=1,
                                       padding=padding_sizes[i])
                maxpool_layer = nn.MaxPool1d(maxpool_size,maxpool_size)
                cnn_model.append(('conv'+str(i+1), conv_layer))
                cnn_model.append(('relu_conv'+str(i+1), nn.ReLU()))
                cnn_model.append(('maxpool'+str(i+1), maxpool_layer))
            seqlen = seqlen // maxpool_size
            
        self.cnn_model = nn.Sequential(OrderedDict(cnn_model))
        
        # LSTM Layers 
        lstm_model = []
        insize = cnn_hidden_sizes[-1]
        lstm_layer = LSTM(insize, lstm_hidden_dim, 
                                num_layers=num_layers, dropout=droprate, 
                                bidirectional=bool(bidirectional))
        lstm_model.append(('lstm', lstm_layer))
        
        self.lstm_layer = nn.Sequential(OrderedDict(lstm_model))
        
        
        # FC Layers
        fc_model = []
        for i in range(len(fc_sizes)):
            if i == 0: 
                insize = lstm_hidden_dim * num_layers 
                if bidirectional: insize *= 2
                fc_layer = nn.Linear(insize,fc_sizes[i])
                
                fc_model.append(('dropout_fc'+str(i+1), nn.Dropout(droprate)))
                fc_model.append(('fc'+str(i+1), fc_layer))
                fc_model.append(('relu_fc'+str(i+1), nn.ReLU()))
            else:
                fc_layer = nn.Linear(fc_sizes[i-1],fc_sizes[i])
                fc_model.append(('dropout_fc'+str(i+1), nn.Dropout(droprate)))
                fc_model.append(('fc'+str(i+1), fc_layer))
                fc_model.append(('relu_fc'+str(i+1), nn.ReLU()))
        
        # Last Layer
        last_layer = nn.Linear(fc_sizes[-1], class_size)
        fc_model.append(('last', last_layer))
        if class_size > 1:
            fc_model.append(('softmax', nn.Softmax(dim=1)))
        self.fc_model = nn.Sequential(OrderedDict(fc_model))
        
    def forward(self, input): 
        cnn_out = self.cnn_model(input.transpose(1,2)) # bs x dim x sl 
        input_t = cnn_out.permute(2,0,1) # transpose to sl x bs x dim
        
        ### TODO: this way of representing the LSTM seems ugly
        lstm_out = self.lstm_layer(input_t)[1][0] # last hidden layer only, discard prev layers and c_n
        lstm_out = lstm_out.transpose(0,1).flatten(start_dim=1) 
        return self.fc_model(lstm_out)
    
    
    # TODO: make a better way to identify the model
    def __repr__(): 
        return 'CNN-LSTM_stacked'