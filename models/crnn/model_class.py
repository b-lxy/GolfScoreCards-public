from pathlib import Path
import os, sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import pandas as pd
from tqdm import tqdm

class CRNN(nn.Module):
    def __init__(self, num_classes, pretrained=None):
        super(CRNN, self).__init__()
        
        if pretrained is not None: # BROKEN
            self.cnn = pretrained['backbone']
            cnn_out_channels = pretrained['channels']
        else:
            # 1. Feature Extraction (CNN): Optimized for 32px height input
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),    # MaxPool only halves the height 28//2=14
                nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),    # height halved again 14//2=7
                nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.Dropout(0.3),
                nn.MaxPool2d((7, 1), (7, 1)), # Keep width resolution by collapsing height
            )
            cnn_out_channels = 256
        
        # 2. Sequence Modeling (RNN)
        self.rnn = nn.LSTM(cnn_out_channels, 128, bidirectional=True, num_layers=1)
        
        # 3. Transcription (Linear Layer to Classify)
        self.fc = nn.Linear(128 * 2, num_classes)  
        # 128 from LSTM's forward pass, 128 for its backward pass

    def forward(self, x):
        # x: [B, 1, 32, 400]
        features = self.cnn(x)              # [B, 256, 1, 100] (assuming 400 width)
        features = F.adaptive_avg_pool2d(features, (1,None))
        features = features.squeeze(2)      # [B, 256, 100]
        features = features.permute(2, 0, 1) # [TimeSteps (100), B, 256]
        output, _ = self.rnn(features)
        logits = self.fc(output)            # [100, B, Num_Classes]
        # CTC Loss expects log_softmax
        return logits.log_softmax(2)
    
class CRNN2(nn.Module):
    def __init__(self, num_classes, pretrained=None):
        super(CRNN2, self).__init__()
        
        if pretrained is not None: # BROKEN
            self.cnn = pretrained['backbone']
            cnn_out_channels = pretrained['channels']
        else:
            # 1. Feature Extraction (CNN): Optimized for 32px height input
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),  # height 32 -> 16, width / 2
                
                nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),  # height halved 32 / 2 = 16
                
                nn.Conv2d(128, 256, 3, 1, 1), nn.GroupNorm(1, 256), nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1), nn.GroupNorm(1, 256), nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),  # height 8 -> 4
                
                nn.Conv2d(256, 512, 3, 1, 1), nn.GroupNorm(1, 512), nn.ReLU(),
                nn.Conv2d(512, 512, 3, 1, 1), nn.GroupNorm(1, 512), nn.ReLU(),
                
                nn.MaxPool2d((4, 1), (4, 1)), # Keep width resolution by collapsing height
                nn.Dropout(0.3)
            )
            cnn_out_channels = 512

        self.layer_norm = nn.LayerNorm(cnn_out_channels)
        
        # 2. Sequence Modeling (RNN)
        self.rnn = nn.LSTM(cnn_out_channels, 128, bidirectional=True, num_layers=2)
        
        # 3. Transcription (Linear Layer to Classify)
        self.fc = nn.Linear(128 * 2, num_classes)  
        # 128 from LSTM's forward pass, 128 for its backward pass

    def forward(self, x):
        features = self.cnn(x)    # [B, C, H, W]
        features = F.adaptive_avg_pool2d(features, (1,None))
        features = features.squeeze(2)     
        features = features.permute(0, 2, 1)  # [B, W, C]
        features = self.layer_norm(features)
        features = features.permute(1, 0, 2)  # [W, B, C]
        output, _ = self.rnn(features)
        logits = self.fc(output)      
        return logits.log_softmax(2)