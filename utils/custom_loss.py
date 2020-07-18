import torch 
import numpy as np

def caution_loss(x, threshold, steepness=1e-7): 
    # steepness is calibrated to have intercept of 0.1 for the dataset
    y = (x-threshold).pow(2) * steepness 
    y[x>threshold] = 0 
    return y

# TODO: combine some codes from here with the one in Balance
class revenue_loss(): 
    def __call__(self, pred_dif, target_dif, y0):
        pred = pred_dif + y0 
        target = target_dif + y0 

        rev = torch.min(pred, target) 
        rev -= torch.nn.functional.relu(pred - target) # adding penalty 

        loss = torch.mean(target - rev) 
        return loss 


class Balance(): 
    def __init__(self, start, reward, fine, unnorm_params, unnorm_kind): 
        self.balance = start 
        self.balance_list = []
        self.diff_list = []
        self.reward = reward 
        self.fine = fine 
        self.unnorm_params = unnorm_params
        self.unnorm_kind = unnorm_kind
        
    def unnormalise(self, a):
        if self.unnorm_kind == 'mean':
            # params = (mean, std)
            return a * self.unnorm_params[1] + self.unnorm_params[0]
        elif self.unnorm_kind == 'minmax': 
            # params = (min, max) 
            return a * (self.unnorm_params[1] - self.unnorm_params[0]) + self.unnorm_params[0]
    
    def update(self, pred, target, y0):
        pred = self.unnormalise(pred.cpu().detach().numpy() + y0.cpu().detach().numpy())
        pred[pred < 0] = 0
        target = self.unnormalise(target.cpu().detach().numpy() + y0.cpu().detach().numpy())
        
        for i in np.arange(len(pred)): 
            self.balance_list.append(float(self.balance))
            diff = float(pred[i] - target[i])
            self.diff_list.append(diff)
            
            # underpredicting
            if diff <= 0: 
                self.balance += pred[i] * self.reward 
            else: # difference > 0 implies overpredicting
                if diff <= self.balance: 
                    self.balance += target[i] * self.reward
                    self.balance -= diff * self.reward
                else: 
                    self.balance += target[i] * self.reward
                    self.balance -= diff * self.fine
    
    def total_profit(self):
        return self.balance - self.balance_list[0]

    
