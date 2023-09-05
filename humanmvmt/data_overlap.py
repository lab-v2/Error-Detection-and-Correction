import os
import pickle
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import random

import torch
from torch.utils.data import DataLoader

from humanmvmt.utils import Dataset


class Overlap_dataloader_1:
    def __init__(self, data_path, overlap_type, sample_type, batch_size = 32, train_shuffle = True):
        self.batch_size = batch_size
        self.data_path = data_path
        self.train_shuffle = train_shuffle
        self.overlap_type = overlap_type
        self.sample_type = sample_type

    def load(self, ):
        filename = os.path.join(self.data_path, f'train_valid_test_{self.overlap_type}_{self.sample_type}.pickle')
        with open(filename, 'rb') as f:
            #train_data, valid_data, test_data = pickle.load(f, encoding='latin1')
            X_train, y_train, X_valid, y_valid, X_test, y_test = pickle.load(f, encoding='latin1')

        #print(train_data[0].shape)
        #print(type(train_data[0]))
        #print(train_data[1].shape)
        #print(type(train_data[1]))
        #from collections import Counter
        #print(Counter(train_data[1]))
        #train_data[0] = np.transpose(train_data[0], (0, 2, 1))
        #test_data[0] = np.transpose(test_data[0], (0, 2, 1))
        #valid_data[0] = np.transpose(valid_data[0], (0, 2, 1))

        #print(train_data[0].shape)
        #print(type(train_data[0]))
        #print(train_data[1].shape)
        #print(type(train_data[1]))
        [a1, b1, c1, d1] = X_train.shape
        X_train = np.reshape(X_train,(-1, c1, d1))
        y_train = np.reshape(y_train, (-1,))

        X_test = np.reshape(X_test,(-1, c1, d1))
        y_test = np.reshape(y_test, (-1,))

        X_valid = np.reshape(X_valid,(-1, c1, d1))
        y_valid = np.reshape(y_valid, (-1,))

        tr_data = Dataset(X_train, y_train)
        te_data = Dataset(X_test, y_test)
        va_data = Dataset(X_valid, y_valid)

        # Create DataLoader for batching
        train_loader = DataLoader(tr_data, batch_size=self.batch_size, shuffle=self.train_shuffle)
        test_loader = DataLoader(te_data, batch_size=self.batch_size, shuffle=False)
        valid_loader = DataLoader(va_data, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader, valid_loader

class Overlap_dataloader_5:
    def __init__(self, data_path, overlap_type, sample_type, batch_size = 32, train_shuffle = True):
        self.batch_size = batch_size
        self.data_path = data_path
        self.train_shuffle = train_shuffle
        self.overlap_type = overlap_type
        self.sample_type = sample_type

    def load(self, ):
        filename = os.path.join(self.data_path, f'train_valid_test_{self.overlap_type}_{self.sample_type}.pickle')
        with open(filename, 'rb') as f:
            #train_data, valid_data, test_data = pickle.load(f, encoding='latin1')
            X_train, y_train, X_valid, y_valid, X_test, y_test = pickle.load(f, encoding='latin1')

        #print(train_data[0].shape)
        #print(type(train_data[0]))
        #print(train_data[1].shape)
        #print(type(train_data[1]))
        #from collections import Counter
        #print(Counter(train_data[1]))
        #train_data[0] = np.transpose(train_data[0], (0, 2, 1))
        #test_data[0] = np.transpose(test_data[0], (0, 2, 1))
        #valid_data[0] = np.transpose(valid_data[0], (0, 2, 1))

        #print(train_data[0].shape)
        #print(type(train_data[0]))
        #print(train_data[1].shape)
        #print(type(train_data[1]))
        print(X_train.shape)
        tr_data = Dataset(X_train, y_train)
        te_data = Dataset(X_test, y_test)
        va_data = Dataset(X_valid, y_valid)

        # Create DataLoader for batching
        train_loader = DataLoader(tr_data, batch_size=self.batch_size, shuffle=self.train_shuffle)
        test_loader = DataLoader(te_data, batch_size=self.batch_size, shuffle=False)
        valid_loader = DataLoader(va_data, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader, valid_loader
    


class LRCNDataLoader:
    def __init__(self, data_path, batch_size = 32, train_shuffle = True, train_split = 0.7):
        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.train_split = train_split
        self.data_path = data_path

    def load(self, ):
        lrcn_tensor = np.load(os.path.join(self.data_path, 'LRCNTensor.npy'))
        lrcn_tensor_labels = np.load(os.path.join(self.data_path, 'LRCNTensorLabels.npy'))

        from collections import Counter
        print(Counter(lrcn_tensor_labels))

        # Transpose to bring it to right shape
        print(lrcn_tensor.shape)
        print(lrcn_tensor_labels.shape)
        #[m, n, mm, nn] = lrcn_tensor.shape
        #new_labels = []
        #new_tensor = []
        #for i in range(m):
        #    for j in range(n):
        #        if lrcn_tensor_labels[i,j] != 6.0:
        #            new_labels.append(lrcn_tensor_labels[i,j])
        #            new_tensor.append(lrcn_tensor[i,j])
        #new_sf = list(zip(new_labels, new_tensor))
        #random.seed(4)
        #random.shuffle(new_sf)
        #new_labels, new_tensor = zip(*new_sf)
        #lrcn_tensor_labels = np.array(new_labels).reshape(-1,n)
        #lrcn_tensor = np.array(new_tensor).reshape(-1,5,mm,nn)
        lrcn_tensor = np.transpose(lrcn_tensor, (0, 1, 3, 2))
        print(lrcn_tensor.shape)
        print(lrcn_tensor_labels.shape)

        # Create Tensor Dataset using the Custom Dataset Class
        dataset = Dataset(lrcn_tensor, lrcn_tensor_labels)

        # Set Train Test Split
        TRAIN_SPLIT = self.train_split
        TEST_SPLIT = 1 - TRAIN_SPLIT
        train_size = int(lrcn_tensor_labels.shape[0] * self.train_split)
        valid_size = int(lrcn_tensor_labels.shape[0] * (self.train_split + 0.1))
        tr_data = torch.utils.data.Subset(dataset, range(train_size))
        va_data = torch.utils.data.Subset(dataset, range(train_size, valid_size))
        te_data = torch.utils.data.Subset(dataset, range(valid_size, lrcn_tensor_labels.shape[0]))

        # Create Train Test Split from dataset with seed 42 ---> Number of sequences x 5 segments x 4 channels x 40 GPS timestamps 
        #tr_data, te_data = torch.utils.data.random_split(dataset, [TRAIN_SPLIT, TEST_SPLIT], 
        #                                                 generator=torch.Generator().manual_seed(42))
        #te_data, va_data = torch.utils.data.random_split(te_data, [0.7, 0.3],
        #                                                generator = torch.Generator().manual_seed(42))

        # Create DataLoader for batching
        train_loader = DataLoader(tr_data, batch_size=self.batch_size, shuffle=self.train_shuffle)
        test_loader = DataLoader(te_data, batch_size=self.batch_size, shuffle=False)
        valid_loader = DataLoader(va_data, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader, valid_loader
