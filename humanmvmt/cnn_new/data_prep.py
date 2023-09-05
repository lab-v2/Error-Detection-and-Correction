import os
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from humanmvmt.utils import Dataset


class CNNDataPrep:
    def __init__(self, data_path, save_path, segment_size = 40):
        self.df = pd.read_pickle(data_path)
        self.save_path = save_path
        self.segment_size = segment_size

    def padding_data(self, ):
        data = np.zeros((self.df['segment_id'].nunique()*self.segment_size, 4))
        label_array = []
        k = 0

        for i, gdf in tqdm(enumerate(self.df.groupby('segment_id'))):
            temp = gdf[1][['velocity', 'acceleration', 'jerk', 'bearing']].values
            label = gdf[1]['label'].iloc[0]
            data[i+k:i+self.segment_size+k] = np.append(temp, np.zeros((self.segment_size - temp.shape[0], 4)), axis=0)
            label_array.append(label)
            k+=(self.segment_size-1)

        np.save(os.path.join(self.save_path, 'CNNData'), data)
        np.save(os.path.join(self.save_path, 'CNNLabels'), np.array(label_array))


class CNNDataLoader:
    def __init__(self, data_path, segment_size, batch_size = 32, train_shuffle = True, train_split = 0.7):
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.train_split = train_split
        self.data_path = data_path

    def load(self, ):
        cnn_data = np.load(os.path.join(self.data_path, 'CNNData.npy'))
        cnn_label = np.load(os.path.join(self.data_path, 'CNNLabels.npy'))

        cnn_3d_array = np.zeros((cnn_data.shape[0]//self.segment_size, self.segment_size, 4))

        for idx, i in enumerate(range(0, cnn_data.shape[0], self.segment_size)):
            cnn_3d_array[idx] = cnn_data[i:i+self.segment_size]

        # Transpose to bring it to right shape (N, LengthSignal, Channels) --> (N, Channels, LengthSignal)
        cnn_3d_array = np.transpose(cnn_3d_array, (0, 2, 1))

        # Create Tensor Dataset using the Custom Dataset Class
        print(cnn_3d_array.shape)
        print(cnn_label.shape)
        print(type(cnn_3d_array))
        print(type(cnn_label))
        dataset = Dataset(cnn_3d_array, cnn_label)

        # Set Train Test Split
        TRAIN_SPLIT = self.train_split
        TEST_SPLIT = 1 - TRAIN_SPLIT

        # Create Train Test Split from dataset with seed 42 ---> Number of sequences x 5 segments x 4 channels x 40 GPS timestamps 
        tr_data, te_data = torch.utils.data.random_split(dataset, [TRAIN_SPLIT, TEST_SPLIT], generator=torch.Generator().manual_seed(42))
        te_data, va_data = torch.utils.data.random_split(te_data, [0.7, 0.3], generator=torch.Generator().manual_seed(42))

        # Create DataLoader for batching
        train_loader = DataLoader(tr_data, batch_size=self.batch_size, shuffle=self.train_shuffle)
        valid_loader = DataLoader(va_data, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(te_data, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader, valid_loader
