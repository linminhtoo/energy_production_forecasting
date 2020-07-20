import torch 
import math
import numpy as np

# TODO: combine some codes from here with the one in Balance
class opportunity_loss(): 
    def __init__(self):
        self.fine = 10
        
    def __call__(self, pred_dif, target_dif, y0, start_bal, overfit_ratio=1, use_gpu=True):
        pred = pred_dif + y0 
        target = target_dif + y0 
        
        if use_gpu: # need to .cuda() relevant tensors 
            rev = torch.min(pred, target).cuda()
            rev -= torch.min(torch.nn.functional.relu(pred - target), 
                             torch.Tensor([start_bal]).cuda().expand_as(pred)) * overfit_ratio # adding penalty  
        else: # not using gpu 
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
            
            # underpredicting
            if diff <= 0: 
                self.balance += pred[i] * self.reward 
            else: # difference > 0 implies overpredicting
                if diff * self.reward * overfit_ratio <= self.balance:  # corrected condition to check bankruptcy (need to multiply diff by price) 
                    self.balance += target[i] * self.reward 
                    self.balance -= diff * self.reward  * overfit_ratio 
                else: # bankruptcy
                    purchasable = math.floor(self.balance / (self.reward * overfit_ratio)) # e.g. if balance = 100, reward = 10, overfit = 2, returns 5 purchasable units
                    self.balance -= purchasable * self.reward * overfit_ratio # deduct cost of purchasable units 
                    self.balance += target[i] * self.reward
                    self.balance -= (diff - purchasable) * self.fine 
                    # (diff - purchasable) is the amount we couldn't buy due to insufficient balance, have to buy it at fine price
           
            self.balance_list.append(float(self.balance))
    
    def total_profit(self):
        return self.balance - self.balance_list[0]

    
