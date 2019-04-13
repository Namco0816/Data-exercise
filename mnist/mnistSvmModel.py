import torch
import torch.nn as nn

class MnistSvmModel(nn.Module):
    def __init__(self):
        super(MnistSvmModel, self).__init__()
        self.feature = nn.Linear(28*28, 1)
    def forward(self,x):
        x = x.view(-1, 28*28)
        x = self.feature(x)
        x = x.view(x.size(0))
        return x
