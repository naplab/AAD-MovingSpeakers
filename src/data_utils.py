import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SeparationDataset(Dataset):
    
    def __init__(self, path):
        super(SeparationDataset, self).__init__()

        self.h5pyLoader = h5py.File(path, 'r')
        
        self.s1 = self.h5pyLoader['spk1']
        self.s2 = self.h5pyLoader['spk2']
        self.noise = self.h5pyLoader['noise']
        
        self._len = self.s1.shape[0]
    
    def __getitem__(self, index):
        
        s1_tensor = torch.from_numpy(self.s1[index].astype(np.float32))
        s2_tensor = torch.from_numpy(self.s2[index].astype(np.float32))
        noise_tensor = torch.from_numpy(self.noise[index].astype(np.float32))  
       
        return s1_tensor, s2_tensor, noise_tensor
    
    def __len__(self):
        return self._len
    
    
class EnhancementDataset(Dataset):
    
    def __init__(self, path):
        super(EnhancementDataset, self).__init__()

        self.h5pyLoader = h5py.File(path, 'r')
        
        self.s1 = self.h5pyLoader['spk1']
        self.s2 = self.h5pyLoader['spk2']
        self.est_s1 = self.h5pyLoader['est_spk1']
        self.est_s2 = self.h5pyLoader['est_spk2']
        self.noise = self.h5pyLoader['noise']
        
        self._len = self.s1.shape[0]
    
    def __getitem__(self, index):

        s1_tensor = torch.from_numpy(self.s1[index].astype(np.float32))
        s2_tensor = torch.from_numpy(self.s2[index].astype(np.float32))
        est_s1_tensor = torch.from_numpy(self.est_s1[index].astype(np.float32))
        est_s2_tensor = torch.from_numpy(self.est_s2[index].astype(np.float32))
        noise_tensor = torch.from_numpy(self.noise[index].astype(np.float32))
        return s1_tensor, s2_tensor, est_s1_tensor, est_s2_tensor, noise_tensor
    
    def __len__(self):
        return self._len
    

class TrajectoryDataset(Dataset):
    
    def __init__(self, path):
        super(TrajectoryDataset, self).__init__()

        self.h5pyLoader = h5py.File(path, 'r')
        
        self.est_s1 = self.h5pyLoader['est_spk1']
        self.est_s2 = self.h5pyLoader['est_spk2']
        self.trace = self.h5pyLoader['trace']
        
        self._len = self.est_s1.shape[0]
    
    def __getitem__(self, index):

        est_s1_tensor = torch.from_numpy(self.est_s1[index].astype(np.float32))
        est_s2_tensor = torch.from_numpy(self.est_s2[index].astype(np.float32))
        trace_tensor = torch.from_numpy(self.trace[index].astype(np.float32))
        
        return est_s1_tensor, est_s2_tensor, trace_tensor
    
    def __len__(self):
        return self._len