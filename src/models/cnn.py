import torch
import torch.nn as nn
import torch.nn.functional as F

class AcousticCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AcousticCNN, self).__init__()
        # Input shape: (Batch Size, 1, 128, M) where M is time frames
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Convolutional Block 4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Global Average Pooling to handle variable time lengths
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# To test the model output shape
if __name__ == "__main__":
    model = AcousticCNN()
    # Dummy input representing (batch_size=4, channels=1, n_mels=128, time_steps=173)
    dummy_input = torch.randn(4, 1, 128, 173) 
    out = model(dummy_input)
    print("Output shape:", out.shape)
