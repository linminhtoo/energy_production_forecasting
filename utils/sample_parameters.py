import numpy as np 

# TODO: Make this sampling scheme generalizable to any objects 
## (i.e. make it hierarchical) 
class ParamSamples(): 
    def __init__(self, ranges_dict): 
        self.rdict = ranges_dict
        self.current_sample = {}
        self.samples = []
        
    def SampleDepend(self, r, dep):
        if 'max_depth' in dep:
            max_d = self.current_sample[dep['max_depth']]
        if 'consecutive' in dep:
            if max_d > len(r): raise ValueError('depth larger than permissible sizes')
            s_idx = np.random.choice(np.arange(len(r) - max_d + 1)) 
            dep_sample = np.array(r[s_idx: s_idx + max_d])
            dep_sample = np.sort(dep_sample * dep['consecutive']) * dep['consecutive']
        if 'monotonic' in dep:
            dep_sample = np.random.choice(r, max_d, replace=True)
            dep_sample = np.sort(dep_sample * dep['monotonic']) * dep['monotonic']
        return dep_sample
                
    def SampleOne(self): 
        for k,(r, dep) in self.rdict.items():
            if dep is None: 
                self.current_sample[k] = np.random.choice(r)
            else: 
                self.current_sample[k] = self.SampleDepend(r, dep)
        self.samples.append(self.current_sample)
        self.current_sample = {}
        
    def SampleAll(self, N):
        for _ in range(N):
            self.SampleOne()
            
        return self.samples 