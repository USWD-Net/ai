from config import WINDOW_SIZE, NUM_CLASSES
import torch
import torch.nn as nn

class MLP(nn.Module) :
    def __init__(self) :
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(WINDOW_SIZE * 4, WINDOW_SIZE*2),
                                     nn.ReLU(),
                                     nn.Linear(WINDOW_SIZE * 2, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, NUM_CLASSES))
        
    def forward(self, s1, s2, s3, s4) :
        return self.mlp(torch.concat([s1,s2,s3,s4], dim=1))
    
    
class CNN(nn.Module) :
    def __init__(self) :
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # [B, 16, 50]
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # [B, 32, 25]
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # [B, 64, 12]
        )
        
        self.flattened_size = 64 * 12

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )
        
    def forward(self, s1, s2, s3, s4) :
        x = torch.stack([s1, s2, s3, s4], dim=1) # [B, 4, 100]
        x = self.cnn(x)
        out = self.mlp(x)
        return out
    
    
class LSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=5):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(4, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 128),
                                nn.ReLU(),
                                nn.Linear(128, NUM_CLASSES))

    def forward(self, s1, s2, s3, s4):
        x = torch.stack([s1, s2, s3, s4], dim=2) # [B, 100, 4]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        output, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(hn[-1])
        out = self.fc(out)
        return out