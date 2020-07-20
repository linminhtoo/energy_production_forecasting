import torch 
import numpy as np

# TODO: combine some codes from here with the one in Balance
class opportunity_loss(): 
    def __init__(self):
        self.fine = 10
        
    def __call__(self, pred_dif, target_dif, y0, start_bal, overfit_ratio=1):
        pred = pred_dif + y0 
        pred = torch.max(pred, torch.Tensor([0]).expand_as(pred))
        target = target_dif + y0 
        
        rev = torch.min(pred, target)
        rev -= torch.min(torch.nn.functional.relu(pred - target), 
                         torch.Tensor([start_bal]).expand_as(pred)) * overfit_ratio # adding penalty  
        
        rev -= torch.nn.functional.relu(pred - target - start_bal) * self.fine
        
        loss = torch.mean(target - rev) 
        return loss 
    
    def __repr__(self): 
        return 'Opportunity Loss'


class Balance(): 
    def __init__(self, start, reward, fine, unnorm_params, unnorm_kind): 
        self.balance = start 
        self.balance_list = [start]
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
    
    def update(self, pred, target, y0, overfit_ratio=1):
        pred = self.unnormalise(pred.cpu().detach().numpy() + y0.cpu().detach().numpy())
        pred[pred < 0] = 0
        target = self.unnormalise(target.cpu().detach().numpy() + y0.cpu().detach().numpy())
        
        for i in np.arange(len(pred)): 
            diff = float(pred[i] - target[i])
            self.diff_list.append(diff)
            
            self.balance += min(pred[i], target[i]) * self.reward
            if (pred[i] - target[i]) > (self.balance / overfit_ratio): 
                self.balance = (pred[i] - target[i]) * -self.fine  # bankruptcy 
            else: 
                self.balance -= (pred[i] - target[i]) * self.reward * overfit_ratio
           
            self.balance_list.append(float(self.balance))
    
    def total_profit(self):
        return self.balance - self.balance_list[0]

    
