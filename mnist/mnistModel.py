import torch.nn as nn

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1,64,3,stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Dropout(0.5),
            nn.LogSoftmax(dim = 1),
        )
    def forward(self,x):
        x = self.feature(x)
        x = x.view(-1, 7*7*128)
        x = self.fc(x)
        return x
