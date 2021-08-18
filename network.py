import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

class MLPNet(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(MLPNet, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, output_dim)
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x