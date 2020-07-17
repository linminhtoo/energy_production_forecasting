import numpy as np
import torch 
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import sklearn.metrics as metrics
import time
from utils.custom_loss import Balance, revenue_loss

class TrainTest(): 
    # TODO: allow inference of class_size from data
    def __init__(self, model, data, params, class_size=2):
        # hyperparams
        self.batch_size      = int(params['BATCH_SIZE'])
        self.epochs          = params['EPOCHS']
        self.loss            = params['LOSS']
        self.early_stopping  = params['EARLY_STOPPING']
        self.patience        = params['PATIENCE']
        self.min_delta       = params['MIN_DELTA']
        self.use_gpu         = torch.cuda.is_available()
        
        # data 
        dataset = TensorDataset(data[0][0], data[0][1])
        self.trainset = DataLoader(dataset, self.batch_size,
                                shuffle=True, drop_last=True)
        if len(dataset) > 0: 
            self.val = True
            dataset = TensorDataset(data[1][0], data[1][1])
            self.valset = DataLoader(dataset, self.batch_size,
                                    shuffle=True, drop_last=True)
        else: 
            self.val = False 
            
        self.testset = data[2]
        del dataset
        
        # model 
        ## TODO: can make this flexible to other inputs -> make model take dict parameters
        if model.__repr__() == 'CNN_1D': 
            self.model = model(input_shape  = data[0][0].shape, 
                               class_size   = class_size, 
                               hidden_sizes = params['HIDDEN_SIZES'], 
                               kernel_sizes = params['KERNEL_SIZES'], 
                               maxpool_size = params['MAXPOOL'],
                               fc_sizes     = params['FC_SIZES'],
                               droprate     = params['DROPOUT']) 
            self.optimizer = params['OPTIMIZER'](self.model.parameters(),
                                                lr=params['LEARNING_RATE'])
        elif model.__repr__() == 'LSTM_': 
            self.model = model(input_shape   = data[0][0].shape, 
                               class_size    = class_size, 
                               hidden_size   = params['HIDDEN_DIM'],
                               fc_sizes      = params['FC_SIZES'],
                               droprate      = params['DROPOUT'],
                               num_layers    = params['NUM_LAYERS'],
                               bidirectional = params['BIDIRECTIONAL']) 
                                
            self.optimizer = params['OPTIMIZER'](self.model.parameters(),
                                                lr=params['LEARNING_RATE'])
        elif model.__repr__() in ['LSTM-CNN_concat', 'LSTM-CNN_stacked', 'CNN-LSTM_stacked']:
            self.model = model(input_shape        = data[0][0].shape, 
                               class_size         = class_size, 
                               cnn_hidden_sizes   = params['CNN_HIDDEN_SIZES'],
                               kernel_sizes       = params['KERNEL_SIZES'], 
                               maxpool_size       = params['MAXPOOL'],
                               lstm_hidden_dim    = params['LSTM_HIDDEN_DIM'],
                               num_layers         = params['NUM_LAYERS'],
                               bidirectional      = params['BIDIRECTIONAL'],
                               fc_sizes           = params['FC_SIZES'],
                               droprate           = params['DROPOUT']) 
                                
            self.optimizer = params['OPTIMIZER'](self.model.parameters(),
                                                lr=params['LEARNING_RATE'])
        if self.use_gpu: 
            self.model.cuda()
        
        # result
        self.mean_train_loss = []
        self.mean_val_loss   = []
        self.min_val_loss    = np.infty
        self.stats           = {}
        self.predictions     = None
        self.bal             = Balance(1e7, 10, 100, (0,87000), 'minmax')
    
    def train_one(self, data):
        if self.use_gpu: 
            X, y = Variable(data[0].cuda()), Variable(data[1].cuda())
        else: 
            X, y = Variable(data[0]), Variable(data[1])
        
        self.model.zero_grad()
        outputs = self.model(X)

        # Updating balance 
        #self.bal.update(outputs.squeeze(), y.squeeze(), X[:, -1, 1])
        
        ## TODO: make this .long generalizable (MSE_loss is not compatible with long)
        loss = self.loss(outputs.squeeze(), y.squeeze()) #, X[:,-1,1].squeeze()) # last argument is stem 
        
        if self.model.training: 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.use_gpu: 
            return loss.data.cpu()
        else:
            return loss.data 
    
    def train(self): 
        tic = time.time()
        
        wait = 0
        for epoch in np.arange(self.epochs): 
            # training
            train_loss = []
            self.model.train()
            for data in tqdm(self.trainset): train_loss.append(self.train_one(data))
            self.mean_train_loss.append(np.mean(train_loss)) 
            
            if self.val: 
                # validating
                val_loss = []
                self.model.eval()
                for data in tqdm(self.valset): val_loss.append(self.train_one(data))

                # check whether to early_stop
                if self.early_stopping and self.min_val_loss - np.mean(val_loss) < self.min_delta:
                    if self.patience <= wait:
                        print('Early stopped at Epoch: ', epoch)
                        self.stats['epoch_stopped'] = epoch 
                        break 
                    else:
                        wait += 1
                        print('decrease in val loss < min_delta, patience count: ', wait)
                else:
                    wait = 0
                    self.min_val_loss = min(self.min_val_loss, np.mean(val_loss))

                self.mean_val_loss.append(np.mean(val_loss))
            
            # printing 
            print('Epoch: {}, train_loss: {}, valid_loss: {}'.format( \
                    epoch, \
                    np.around(np.mean(train_loss), decimals=8),\
                    np.around(np.mean(val_loss), decimals=8)))
            
        self.stats['train_time'] = time.time() - tic
        return 0 
   
    
    def save_stats(self):
        self.stats['mean_train_loss'] = self.mean_train_loss
        self.stats['mean_val_loss']   = self.mean_val_loss
        self.stats['min_val_loss']    = self.min_val_loss
        self.stats['predictions']     = self.predictions
        self.stats['bal_list']        = np.array(self.bal.balance_list)
        self.stats['revenue']         = self.bal.balance_list[-1]
        
    def test(self): 
        if self.use_gpu: 
            X, y = Variable(self.testset[0].cuda()), Variable(self.testset[1].cuda())
        else: 
            X, y = Variable(self.testset[0]), Variable(self.testset[1])
        
        # make this generalisable 
        outputs  = self.model(X).data.squeeze()
        self.predictions = outputs
        self.stats['test_loss'] = self.loss(self.predictions, y.squeeze())#, X[:,-1,1].squeeze()) 
        # TODO: to incorporate the 3rd argument nicely^
        if self.use_gpu: 
            self.bal.update(outputs.squeeze().cpu(), y.squeeze().cpu(), X[:, -1, 1])
        else: 
            self.bal.update(outputs.squeeze(), y.squeeze(), X[:, -1, 1])
        
        self.save_stats()
        print('train_time: ' + str(self.stats['train_time']))
        print('test_loss: ' + str(self.stats['test_loss']))
        print('revenue: ' + str(self.stats['revenue']/1e7))
        return self.stats 