import torch.nn as nn
from torch import flatten
import torch.nn.functional as F

class CNN_A(nn.Module):
    def __init__(self, num_channels, classes = None, is_lrcn = False):
        super(CNN_A, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = num_channels, out_channels = 32, kernel_size = 3)
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
        if not is_lrcn:
            self.fc1 = nn.Linear(in_features = 32*36, out_features = classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.is_lrcn = is_lrcn       

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = flatten(x, 1)
        if self.is_lrcn:
            return x
        x = self.fc1(x)
        output = self.logSoftmax(x)
        return x, output


class CNN_B(nn.Module):
    def __init__(self, num_channels, classes = None, is_lrcn = False):
        super(CNN_B, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = num_channels, out_channels = 32, kernel_size = 3)
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.conv4 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3)
        if not is_lrcn:
            self.fc1 = nn.Linear(in_features = 64*32, out_features = classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.is_lrcn = is_lrcn        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = flatten(x, 1)
        if self.is_lrcn:
            return x
        x = self.fc1(x)
        output = self.logSoftmax(x)
        return x, output


class CNN_C(nn.Module):
    def __init__(self, num_channels, classes = None, is_lrcn = False):
        super(CNN_C, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = num_channels, out_channels = 32, kernel_size = 3)
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.conv4 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.conv5 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv6 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3)
        if not is_lrcn:
            self.fc1 = nn.Linear(in_features = 128*188, out_features = classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.is_lrcn = is_lrcn       

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = flatten(x, 1)
        if self.is_lrcn:
            return x
        x = self.fc1(x)
        output = self.logSoftmax(x)
        return x, output


class CNN_D(nn.Module):
    def __init__(self, num_channels, classes = None, is_lrcn = False):
        super(CNN_D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = num_channels, out_channels = 32, kernel_size = 3)
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.conv4 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.conv5 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv6 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3)
        self.fc1 = nn.Linear(in_features = 128*28, out_features =  int(128*28*1/4))
        if not is_lrcn:
            self.fc2 = nn.Linear(in_features = int(128*28*1/4), out_features = classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.is_lrcn = is_lrcn       

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = flatten(x, 1)
        if self.is_lrcn:
            return x
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return x, output


class CNN_E(nn.Module):
    def __init__(self, num_channels, classes = None, is_lrcn = False):
        super(CNN_E, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = num_channels, out_channels = 32, kernel_size = 3)
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.conv4 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.conv5 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv6 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3)
        self.fc1 = nn.Linear(in_features = 128, out_features = int(128*1/4))
        if not is_lrcn:
            self.fc2 = nn.Linear(in_features = int(128*1/4), out_features = classes)
        self.max_pool = nn.MaxPool1d(2)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.is_lrcn = is_lrcn       

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.max_pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.max_pool(x)
        x = flatten(x, 1)
        if self.is_lrcn:
            return x
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return x, output


class CNN_F(nn.Module):
    def __init__(self, num_channels, classes = None, is_lrcn = False):
        super(CNN_F, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = num_channels, out_channels = 32, kernel_size = 3)
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.conv4 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.conv5 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv6 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3)
        self.fc1 = nn.Linear(in_features = 128, out_features = int(128*1/4))
        if not is_lrcn:
            self.fc2 = nn.Linear(in_features = int(128*1/4), out_features = classes)
        self.max_pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p = 0.5)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.is_lrcn = is_lrcn       

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.max_pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.max_pool(x)
        x = self.dropout(x)
        x = flatten(x, 1)
        if self.is_lrcn:
            return x
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return x, output


class CNN_G(nn.Module):
    def __init__(self, num_channels, segment_size = 40, classes = None, is_lrcn = False):
        super(CNN_G, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = num_channels, out_channels = 32, kernel_size = 3, padding = 'same')
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 'same')
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 'same')
        self.conv4 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 'same')
        self.conv5 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 'same')
        self.conv6 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 'same')
        self.fc1 = nn.Linear(in_features = 16 * segment_size, out_features = 4 * segment_size)
        if not is_lrcn:
            self.fc2 = nn.Linear(in_features = 4 * segment_size, out_features = classes)
        self.max_pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p = 0.5)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.is_lrcn = is_lrcn       

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.max_pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.max_pool(x)
        x = self.dropout(x)
        x = flatten(x, 1)
        if self.is_lrcn:
            return x
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return x, output


class CNN_H(nn.Module):
    def __init__(self, num_channels, classes = None, is_lrcn = False):
        super(CNN_H, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = num_channels, out_channels = 32, kernel_size = 3)
        self.conv2 = nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.conv3 = nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3)
        self.conv4 = nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 3)
        self.conv5 = nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3)
        self.conv6 = nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 3)
        self.conv7 = nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 3)
        self.conv8 = nn.Conv1d(in_channels = 256, out_channels = 256, kernel_size = 3)
        self.fc1 = nn.Linear(in_features = 256, out_features = int(256*1/4))
        if not is_lrcn:
            self.fc2 = nn.Linear(in_features = int(256*1/4), out_features = classes)
        self.max_pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p = 0.5)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.is_lrcn = is_lrcn       

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.max_pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.max_pool(x)
        x = self.dropout(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.max_pool(x)
        x = self.dropout(x)
        x = flatten(x, 1)
        if self.is_lrcn:
            return x
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return x, output


class CNN_I:
    def __init__(self):
        pass
