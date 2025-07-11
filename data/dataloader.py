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
        
    def preprocessing(self, data) :
        pass
        
    def __getitem__(self, idx) :
        s1, s2, s3, s4 = self.data[idx] # scaled timeseries data
        lbl = self.labels[idx]
        lbl = torch.tensor(lbl)
        return s1, s2, s3, s4, lbl
    
    def __len__(self) :
        return len(self.data)
        

def get_dataloader(data_root) :
    train_dataset = USWDDataset(os.path.join(data_root, "train/train.pkl"), is_sample=True)
    test_dataset = USWDDataset(os.path.join(data_root, "test/test.pkl"), is_sample=True)
    train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = True)
    return train_dataloader, test_dataloader