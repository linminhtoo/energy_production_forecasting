import torch 
import numpy as np

class opportunity_loss(): 
    def __init__(self):
        self.fine = 10
        
    def __call__(self, pred_dif, target_dif, y0, start_bal, overpred_ratio=1):
        pred = pred_dif + y0 
        if use_gpu:
          pred = torch.max(pred, torch.Tensor([0]).cuda().expand_as(pred)).cuda()
        else:
          pred = torch.max(pred, torch.Tensor([0]).expand_as(pred))
        target = target_dif + y0 
        diff = pred - target 
        
        if use_gpu:
          start_bal = torch.Tensor([start_bal]).cuda().expand(pred.shape[0])
          bal = torch.min(pred, target).cuda() + start_bal 
        else:
          start_bal = torch.Tensor([start_bal]).expand(pred.shape[0])
          bal = torch.min(pred, target) + start_bal          
        
        # conditions
        overpredict = (diff > 0).float()
        can_cover = (diff * overpred_ratio <= bal).float() 
        
        # if overpredict, but can cover 
        bal -= (overpredict * can_cover) * diff * overpred_ratio 
        # if overpredict, but cannot cover 
        if use_gpu:
          buffer = torch.max((overpredict * (1-can_cover)) * bal, torch.zeros(pred.shape[0]).cuda()).cuda()
          bal -= buffer + (overpredict * (1-can_cover)) * (diff * overpred_ratio - bal) / overpred_ratio * self.fine           
        else:
          buffer = torch.max((overpredict * (1-can_cover)) * bal, torch.zeros(pred.shape[0]))
          bal -= buffer + (overpredict * (1-can_cover)) * (diff * overpred_ratio - bal) / overpred_ratio * self.fine 
        
        rev = bal - start_bal 
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
    
    def update(self, pred, target, y0, overpred_ratio=1):
        pred = self.unnormalise(pred.cpu().detach().numpy() + y0.cpu().detach().numpy())
        pred[pred < 0] = 0
        target = self.unnormalise(target.cpu().detach().numpy() + y0.cpu().detach().numpy())
        
        for i in np.arange(len(pred)): 
            diff = float(pred[i] - target[i])
            self.diff_list.append(diff)
            over_pen = self.reward * overpred_ratio 
            
            self.balance += min(pred[i], target[i]) * self.reward
            # overpredicting 
            if diff > 0: 
                if diff * over_pen <= self.balance: 
                    self.balance -= diff * self.reward * overpred_ratio 
                else:
                    self.balance -= (diff * over_pen - self.balance - max(self.balance,0)) / over_pen * self.fine + max(self.balance,0)

            self.balance_list.append(float(self.balance))
    
    def total_profit(self):
        return self.balance - self.balance_list[0]
