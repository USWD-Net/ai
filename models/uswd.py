from config import *
from models.net import *
import torch
import torch.nn as nn


class USWDScorer(nn.Module) :
    def __init__(self) :
        super(USWDScorer, self).__init__()
        
        self.sconv1_residual = nn.Sequential(
            SeparableConv(1, 32),
            nn.LeakyReLU(),
            nn.Dropout2d(p=DROPOUT_SCORE),
            SeparableConv(32, 64),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.sconv1_shortcut = nn.Sequential(
            nn.Conv2d(1, 64, 1, stride=2, padding=0),
            nn.BatchNorm2d(64)
        )

        self.sconv2_residual = nn.Sequential(
            SeparableConv(64, 128),
            nn.LeakyReLU(),
            nn.Dropout2d(p=DROPOUT_SCORE),
            SeparableConv(128, 128),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.sconv2_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2, padding=0),
            nn.BatchNorm2d(128)
        )

        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(128, 4), nn.Dropout(p=DROPOUT_SCORE))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, s1, s2, s3, s4) :
        x = torch.stack([s1, s2, s3, s4], dim=1) # [B, 4, 100]
        x1 = x.unsqueeze(dim=1) # [B, 1, 4, 100]
        x2 = self.sconv1_residual(x1) + self.sconv1_shortcut(x1) # [B, 64, 2, 50]
        x3 = self.sconv2_residual(x2) + self.sconv2_shortcut(x2) # [B, 128, 1, 25]
        out = self.gap(x3) # [B, 128]
        out = self.fc(out) # [B, 4]
        out = self.sigmoid(out)
        return out
    
class USWDFFTEncoder(nn.Module) :
    def __init__(self, emb_dim) :
        super(USWDFFTEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels = 16, out_channels = emb_dim, kernel_size = 3, stride = 2),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x) :
        x = x.unsqueeze(dim=1)
        x = self.encoder(x)
        out = self.gap(x)
        return out.squeeze(dim=-1)

class USWDSignalTransformerEncoder(nn.Module) :
    def __init__(self, d_model) :
        super(USWDSignalTransformerEncoder, self).__init__()
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels = 1, out_channels = d_model, kernel_size = 3, stride = 2),
            nn.ReLU())
        
        self.te = TrasnformerEncoder(d_model = d_model,
                                     nhead = N_HEAD,
                                     num_layers = NUM_LAYERS,
                                     dropout = DROPOUT)
        
    def forward(self, x) :
        x = x.unsqueeze(dim=1)
        x = self.cnn_encoder(x)
        x = self.te(x)
        return x
    
class USWDSignalCNNEncoder(nn.Module) :
    def __init__(self, emb_dim) :
        super(USWDSignalCNNEncoder, self).__init__()
        self.sconv1_residual = nn.Sequential(
            SeparableConv1d(1, 32),
            nn.LeakyReLU(),
            SeparableConv1d(32, 64),
            nn.MaxPool1d(3, stride=2, padding=1)
        )

        self.sconv1_shortcut = nn.Sequential(
            nn.Conv1d(1, 64, 1, stride=2, padding=0),
            nn.BatchNorm1d(64)
        )

        self.sconv2_residual = nn.Sequential(
            SeparableConv1d(64, 128),
            nn.LeakyReLU(),
            SeparableConv1d(128, 128),
            nn.MaxPool1d(3, stride=2, padding=1)
        )

        self.sconv2_shortcut = nn.Sequential(
            nn.Conv1d(64, 128, 1, stride=2, padding=0),
            nn.BatchNorm1d(128)
        )

        self.sconv3_residual = nn.Sequential(
            SeparableConv1d(128, 256),
            nn.LeakyReLU(),
            SeparableConv1d(256, 256),
            nn.MaxPool1d(3, stride=2, padding=1)
        )

        self.sconv3_shortcut = nn.Sequential(
            nn.Conv1d(128, 256, 1, stride=2, padding=0),
            nn.BatchNorm1d(256)
        )

        self.gap = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.fc = nn.Linear(256, emb_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x) :
        x1 = x.unsqueeze(dim=1) # [B, 1, 100]
        x2 = self.sconv1_residual(x1) + self.sconv1_shortcut(x1)
        x3 = self.sconv2_residual(x2) + self.sconv2_shortcut(x2)
        x4 = self.sconv3_residual(x3) + self.sconv3_shortcut(x3)
        print(x4.shape)
        out = self.gap(x4)
        print(out.shape)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
        

class USWDEncoder(nn.Module) :
    def __init__(self, emb_dim, out_dim, encoder_type = "c") :
        super(USWDEncoder, self).__init__()
        self.fft_encoder = USWDFFTEncoder(emb_dim)
        self.fft_linear = nn.Linear(emb_dim, out_dim)
        if encoder_type == "c" :
            self.signal_encoder = USWDSignalCNNEncoder(emb_dim)
        elif encoder_type == "t" :
            self.signal_encoder = USWDSignalTransformerEncoder(d_model = emb_dim)
        self.signal_linear = nn.Linear(emb_dim, out_dim)

    def forward(self, x) :
        # x [B, 100]
        x_fft = torch.abs(torch.fft.rfft(x, n=WINDOW_SIZE)) # x_fft [B, 51]
        emb_x_fft = self.fft_encoder(x_fft) # [B, 128]
        emb_x_fft = self.fft_linear(emb_x_fft) # [B, 256]
        emb_x = self.signal_encoder(x) # [B, ]
        emb_x = self.signal_linear(emb_x)
        
        out = emb_x_fft + emb_x
        return out
        
        
    
class USWDNet_Sep(nn.Module) :
    def __init__(self) :
        super(USWDNet_Sep, self).__init__()
        self.scorer = USWDScorer()
        self.s1_encoder = USWDEncoder(emb_dim = EMB_DIM, out_dim = OUT_DIM, encoder_type=ENCODER_TYPE)
        self.s2_encoder = USWDEncoder(emb_dim = EMB_DIM, out_dim = OUT_DIM, encoder_type=ENCODER_TYPE)
        self.s3_encoder = USWDEncoder(emb_dim = EMB_DIM, out_dim = OUT_DIM, encoder_type=ENCODER_TYPE)
        self.s4_encoder = USWDEncoder(emb_dim = EMB_DIM, out_dim = OUT_DIM, encoder_type=ENCODER_TYPE)
        self.cls_header = nn.Sequential(
            nn.Linear(OUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES),
            nn.Sigmoid()
        )
        
    def forward(self, s1, s2, s3, s4) :
        emb_s1 = self.s1_encoder(s1) # [B, 256]
        emb_s2 = self.s2_encoder(s2) # [B, 256]
        emb_s3 = self.s3_encoder(s3) # [B, 256]
        emb_s4 = self.s4_encoder(s4) # [B, 256]
        weights = self.scorer(s1, s2, s3, s4).unsqueeze(2) # [B, 4, 1]
        
        embs = torch.stack([emb_s1, emb_s2, emb_s3, emb_s4], dim=1)
        w_embs = embs * weights
        w_sum = torch.sum(w_embs, dim = 1) # [B, 256]
        out = self.cls_header(w_sum)
        return out
    
class USWDNet_Uni(nn.Module) :
    def __init__(self) :
        super(USWDNet_Uni, self).__init__()
        self.scorer = USWDScorer()
        self.encoder = USWDEncoder(emb_dim = EMB_DIM, out_dim = OUT_DIM, encoder_type=ENCODER_TYPE)
        self.cls_header = nn.Sequential(
            nn.Linear(OUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES),
            nn.Sigmoid()
        )
        
    def forward(self, s1, s2, s3, s4) :
        emb_s1 = self.encoder(s1) # [B, 256]
        emb_s2 = self.encoder(s2) # [B, 256]
        emb_s3 = self.encoder(s3) # [B, 256]
        emb_s4 = self.encoder(s4) # [B, 256]
        weights = self.scorer(s1, s2, s3, s4).unsqueeze(2) # [B, 4, 1]
        
        embs = torch.stack([emb_s1, emb_s2, emb_s3, emb_s4], dim=1)
        w_embs = embs * weights
        w_sum = torch.sum(w_embs, dim = 1) # [B, 256]
        out = self.cls_header(w_sum)
        return out # [B, N_CLS]