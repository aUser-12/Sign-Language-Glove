
import torch
import torch.nn as nn

class FlexGloveCNN(nn.Module):
    """
    1D CNN classifier for flex glove gestures.
    Input:  (batch, 1, window_size * num_sensors) flattened time window
    Output: (batch, num_classes) logits
    """
    def __init__(self, num_sensors: int, window_size: int, num_classes: int):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size

       
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=num_sensors, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8), 
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
       
        x = x.permute(0, 2, 1) 
      
        x = self.conv_block(x)
        return self.classifier(x)
