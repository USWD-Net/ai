from torch.utils.data import Dataset, DataLoader

class USWDTrainDataset(Dataset) :
    def __init__(self, data) :
        super(USWDTrainDataset, self).__init__()
        
        
    def __getitem__(self, idx) :
        return None
    
    def __len__(self) :
        return None
        
class USWDTestDataset(Dataset) :
    def __init__(self, data) :
        super(USWDTestDataset, self).__init__()
        
        
    def __getitem__(self, idx) :
        return None
    
    def __len__(self) :
        return None
               

def get_dataloader(is_val = False) :
    pass