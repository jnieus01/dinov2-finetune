import torch
import torch.nn as nn

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

class DINOLinearClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DINOLinearClassifier, self).__init__()
        self.model = model
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
        x = self.fc(x)
        return x