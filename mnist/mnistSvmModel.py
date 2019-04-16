import torch
import torch.nn as nn

class MnistSvmModel(nn.Module):
    def __init__(self):
        super(MnistSvmModel, self).__init__()
        self.feature = nn.Linear(28*28, 10)
        self.classifier = nn.LogSoftmax()
    def forward(self,x):
        x = x.view(-1, 28*28)
        x = self.feature(x)
        x = self.classifier(x)
        return x
