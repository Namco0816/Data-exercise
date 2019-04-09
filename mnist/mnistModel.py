import torch.nn as nn

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU(),
            nn.LogSoftmax(dim = 1),
        )
    def forward(self,x):
        x = x.view(-1,28*28)
        x = self.feature(x)
        return x
