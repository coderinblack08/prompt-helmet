import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import math


class BaseClassifier(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_name(self):
        pass

class SimpleCNNClassifier(BaseClassifier):
    def __init__(self, input_shape):
        super().__init__()
        self.name = "simple_cnn"
        
        # Calculate optimal number of pooling layers based on input size
        h, w = input_shape
        n_pools = min(
            3,  # Don't pool more than 3 times
            min(
                int(math.log2(h/2)),
                int(math.log2(w/2))
            )
        )
        
        # Build CNN layers dynamically
        layers = []
        in_channels = 1
        channels = [16, 32, 64][:n_pools + 1]  # Adapt channel progression to n_pools
        
        for i in range(n_pools):
            layers.extend([
                nn.Conv2d(in_channels, channels[i], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i]),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_channels = channels[i]
        
        # Add final conv block without pooling
        layers.extend([
            nn.Conv2d(in_channels, channels[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU()
        ])
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size
        with torch.no_grad():
            x = torch.zeros(1, 1, input_shape[0], input_shape[1])
            x = self.conv_layers(x)
            flat_size = x.view(1, -1).shape[1]
        
        # Adapt classifier head size based on flattened size
        hidden_size = min(256, flat_size)  # Don't make hidden larger than input
        self.classifier = nn.Sequential(
            nn.Linear(flat_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_name(self):
        return self.name