import torch
import torch.nn as nn
import torch.nn.functional as F

class GTSRB(nn.Module):
    def __init__(self, in_planes = 1, planes = 6, stride=1, mode='train'):
        super(GTSRB, self).__init__()
        self.mode = mode
        self.keep_prob = (0.5 if (mode=='train') else 1.0)
        self.conv1 = nn.Conv2d(3, 32, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, 3, stride = 1, padding = 1)
        self.conv6 = nn.Conv2d(128, 128, 3, stride = 1, padding = 1)
        #self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(4*4*128, 512)
        self.dropout = nn.Dropout2d(p=self.keep_prob)
        self.fc2 = nn.Linear(512, 43)
        #self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        #x = self.fc3(x)
        return x

    def penultimate(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x
        
def gtsrb():
    return GTSRB()

