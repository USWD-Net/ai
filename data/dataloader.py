import random
from config import *
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader

class USWDDataset(Dataset) :
    def __init__(self, data_path, is_sample=True) :
        super(USWDDataset, self).__init__()
        self.is_sample = is_sample
        self.data = []
        self.labels = []
        self.load_data(data_path=data_path)

    def load_data(self, data_path) :
        with open(data_path, "rb") as f :
            d = pickle.load(f)
            
        if self.is_sample :
            tmp_lbl_0 = random.sample(d[0], len(d[1]))
            
        for i in tmp_lbl_0 :
            self.data.append(i)
            self.labels.append(0)
        
        for i in list(d.keys()) :
            if i != 0 :
                for a in d[i] :
                    self.data.append(a)
                    self.labels.append(i)
        del d
        
        
    def __getitem__(self, idx) :
        s1, s2, s3, s4 = self.data[idx] # scaled timeseries data
        s1 = torch.tensor(s1, dtype=torch.float32)
        s2 = torch.tensor(s2, dtype=torch.float32)
        s3 = torch.tensor(s3, dtype=torch.float32)
        s4 = torch.tensor(s4, dtype=torch.float32) # (batch, window size) float tensor 
            
        lbl = self.labels[idx]
        lbl = torch.tensor(lbl, dtype=torch.float32) # (batch, 1) float tensor
        return s1, s2, s3, s4, lbl
    
    def __len__(self) :
        return len(self.data)
        

# rfft 하면 - 51?

def get_dataloader(data_root) :
    train_dataset = USWDDataset(os.path.join(data_root, "train.pkl"), is_sample=True)
    val_dataset = USWDDataset(os.path.join(data_root, "val.pkl"), is_sample=True)
    test_dataset = USWDDataset(os.path.join(data_root, "test.pkl"), is_sample=True)
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = True)
    return train_dataloader, val_dataloader, test_dataloader