import os
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from humanmvmt.utils import Dataset


class LRCNDataPrep:
    def __init__(self, data_path, save_path, segment_size = 40):
        self.df = pd.read_pickle(data_path)
        self.save_path = save_path
        self.segment_size = segment_size

    def padding_data(self, ):
        # Normalizing Segments from Dataframe (Equal sized Segments)
        data = np.zeros((self.df['segment_id'].nunique()*self.segment_size, 6))
        label_array = []
        k = 0

        for i, gdf in tqdm(enumerate(self.df.groupby('segment_id'))):
            seg_id, trip_id, label = gdf[1]['segment_id'].iloc[0], gdf[1]['trip_id'].iloc[0], gdf[1]['label'].iloc[0]
            temp = gdf[1][['velocity', 'acceleration', 'jerk', 'bearing', 'segment_id', 'trip_id']].values

            pad_array = np.zeros((self.segment_size - temp.shape[0], 4))

            pad_array_sid_col = np.ones((self.segment_size - temp.shape[0]))*seg_id
            pad_array_sid_col = pad_array_sid_col.reshape(self.segment_size - temp.shape[0], 1)
            pad_array = np.hstack((pad_array, pad_array_sid_col))

            pad_array_tid_col = np.ones((self.segment_size - temp.shape[0]))*trip_id
            pad_array_tid_col = pad_array_tid_col.reshape(self.segment_size - temp.shape[0], 1)
            pad_array = np.hstack((pad_array, pad_array_tid_col))

            data[i+k:i+self.segment_size+k] = np.append(temp, pad_array, axis=0)
            label_array.append(label)
            k+=(self.segment_size-1)

        np.save(os.path.join(self.save_path, 'LRCNData'), data)
        np.save(os.path.join(self.save_path, 'LRCNLabels'), np.array(label_array))

    def create_tensor(self, sequence_length = 5):
        # Create LRCN Tensor
        lrcn_data = np.load(os.path.join(self.save_path, 'LRCNData.npy'))
        lrcn_label = np.load(os.path.join(self.save_path, 'LRCNLabels.npy'))

        # Checks
        assert lrcn_data.shape[0] == np.repeat(lrcn_label, self.segment_size).shape[0]

        # Stack labels to the data 
        lrcn_data = np.hstack((lrcn_data, np.repeat(lrcn_label, 40).reshape(-1, 1)))

        ## Convert to Tensor
        lrcn_tensor = np.zeros((1, sequence_length, self.segment_size, 4))
        lrcn_label_array = []

        for tid in tqdm(np.unique(lrcn_data[:, 5])):
            trip_fltr = np.asarray([tid])
            tid_array = lrcn_data[np.in1d(lrcn_data[:, 5], trip_fltr)]  
            seg_ids = np.unique(tid_array[:, 4])    

            if len(seg_ids) >= 2:
                # Create sequences
                sequence = []
                if len(seg_ids) >= sequence_length:
                    for i in range(len(seg_ids) - sequence_length + 1):
                        sequence.append(seg_ids[i: i + sequence_length])
                else:
                    sequence = [seg_ids]

                # Add each segment present in the sequence in numpy tensor
                seq_tensor = np.zeros((len(sequence), sequence_length, self.segment_size, 4))
                for idx, seq in enumerate(sequence):
                    sequence_labels = []
                    for i in range(sequence_length):
                        try:
                            seg = seq[i]
                            
                            seg_fltr = np.asarray([seg])
                            sid_array = tid_array[np.in1d(tid_array[:, 4], seg_fltr)]  
                            curr_label = np.unique(sid_array[:, 6])
                            sequence_labels.append(curr_label[0])
                            seq_tensor[idx][i] = sid_array[:, :4]
                        except:
                            seq_tensor[idx][i] = np.zeros((self.segment_size, 4))
                            sequence_labels.append(6.0) # Append dummy label 6
                    lrcn_label_array.append(sequence_labels)

                lrcn_tensor = np.append(lrcn_tensor, seq_tensor, axis=0)

        np.save(os.path.join(self.save_path, 'LRCNTensor'), lrcn_tensor[1:]) #Remove the first dummy inclusion
        np.save(os.path.join(self.save_path, 'LRCNTensorLabels'), np.array(lrcn_label_array))


class LRCNDataLoader:
    def __init__(self, data_path, batch_size = 32, train_shuffle = True, train_split = 0.7):
        self.batch_size = batch_size
        self.train_shuffle = train_shuffle
        self.train_split = train_split
        self.data_path = data_path

    def load(self, ):
        lrcn_tensor = np.load(os.path.join(self.data_path, 'LRCNTensor.npy'))
        lrcn_tensor_labels = np.load(os.path.join(self.data_path, 'LRCNTensorLabels.npy'))

        # Transpose to bring it to right shape
        lrcn_tensor = np.transpose(lrcn_tensor, (0, 1, 3, 2))

        # Create Tensor Dataset using the Custom Dataset Class
        dataset = Dataset(lrcn_tensor, lrcn_tensor_labels)

        # Set Train Test Split
        TRAIN_SPLIT = self.train_split
        TEST_SPLIT = 1 - TRAIN_SPLIT

        # Create Train Test Split from dataset with seed 42 ---> Number of sequences x 5 segments x 4 channels x 40 GPS timestamps 
        tr_data, te_data = torch.utils.data.random_split(dataset, [TRAIN_SPLIT, TEST_SPLIT], 
                                                         generator=torch.Generator().manual_seed(42))
        te_data, va_data = torch.utils.data.random_split(te_data, [0.5, 0.5],
                                                        generator = torch.Generator().manual_seed(42))

        # Create DataLoader for batching
        train_loader = DataLoader(tr_data, batch_size=self.batch_size, shuffle=self.train_shuffle)
        test_loader = DataLoader(te_data, batch_size=self.batch_size, shuffle=False)
        valid_loader = DataLoader(va_data, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader, valid_loader
