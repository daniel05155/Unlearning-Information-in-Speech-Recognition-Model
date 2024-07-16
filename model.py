import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_class):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1,out_channels=8,kernel_size=13,stride=1)
        # self.dropout1 = nn.Dropout(0.3) 
        self.conv2 = nn.Conv1d(in_channels=8,out_channels=16,kernel_size=11,stride=1)
        # self.dropout2 = nn.Dropout(0.3)
        self.conv3 = nn.Conv1d(in_channels=16,out_channels=32,kernel_size=9,stride=1)
        # self.dropout3 = nn.Dropout(0.3)
        self.conv4 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=7,stride=1)
        # self.dropout4 = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(12416, 256)
        self.dropout5 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256,128)
        self.dropout6 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_class)
        
    def forward(self, x):
        x = F.max_pool1d(F.relu(self.conv1(x)),kernel_size=3)
        # x = self.dropout1(x)
        x = F.max_pool1d(F.relu(self.conv2(x)),kernel_size=3)
        # x = self.dropout2(x)
        x = F.max_pool1d(F.relu(self.conv3(x)),kernel_size=3)
        # x = self.dropout3(x)
        x = F.max_pool1d(F.relu(self.conv4(x)),kernel_size=3)
        # x = self.dropout4(x)
        x = F.relu(self.fc1(x.reshape(-1,x.shape[1] * x.shape[2])))
        x = self.dropout5(x)
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        x = self.fc3(x)
        #print(x.shape)
        return x 

if __name__ == '__main__':
    x = torch.rand([1,128,81])
    print(x.shape)
    model = CNN(35)
    out = model(x)