import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import permutations
from humanmvmt.cnn.data_prep import CNNDataLoader
from humanmvmt.trainer import Trainer


class Introspection:
    def __init__(self, feature_array_path, label_array_path, distance_function, segment_size = 40, num_channels = 4) -> None:
        # Load Data
        self.data = np.load(feature_array_path)
        self.label = np.load(label_array_path)
        self.segment_size = segment_size
        self.num_channels = num_channels
        self.distance_function = distance_function
    
    def get_predicted_values(self, pred_data_path, model_checkpoint):
        te_loader = CNNDataLoader(pred_data_path, segment_size = self.segment_size, batch_size = 64, train_shuffle=False, 
                                  train_split = 1) #keep_original_order=True)
        pred_loader, _ = te_loader.load()

        tr_load = Trainer()
        self.pred_label, y_true = tr_load.predict(pred_loader, saved_model_path = model_checkpoint)
        print("Inference on model object completed")

    def generate_rule_learning_db(self, inequality = '>', prefix = 'd', feature_names = None, label_dict = None, class_thresholds = None):
        # Convert to 3d Tensors
        self.tensor_data = np.zeros((self.data.shape[0]//self.segment_size, self.segment_size, self.num_channels))

        for idx, i in enumerate(range(0, self.data.shape[0], self.segment_size)):
            self.tensor_data[idx] = self.data[i:i+self.segment_size]

        # Transpose to bring it to right shape (N, LengthSignal, Channels) --> (N, Channels, LengthSignal)
        self.tensor_data = np.transpose(self.tensor_data, (0, 2, 1))

        # Create global array for querying
        self.tensor_data = self.tensor_data.reshape(-1, self.segment_size * self.num_channels)
        self.tensor_data = np.hstack((self.tensor_data, self.label.reshape(-1, 1)))
    
        # Compute Means for Each Travel Mode
        class_samples = {}
        for label in np.unique(self.label):
            fltr = np.asarray([label])
            if label_dict is not None:
                class_samples[label_dict[label]] = self.tensor_data[np.in1d(self.tensor_data[:, self.segment_size * self.num_channels], fltr)]
            else:
                class_samples[label] = self.tensor_data[np.in1d(self.tensor_data[:, self.segment_size * self.num_channels], fltr)]

        final_list = []
        current_vector_label = -1

        # Iterate through each sample
        for idx, sample in tqdm(enumerate(self.tensor_data, start=1)):
            sample_atoms = []

            # Iterate through each feature vector sample
            feature_vector_splits = [(i, i+self.segment_size) for i in range(0, self.segment_size*self.num_channels, 40)]
            for low, up in feature_vector_splits: 

                # Compute Means
                if current_vector_label != low:
                    label_mean_dict = {label:np.mean(class_vector[:, low:up], axis=0).reshape(1, -1) \
                                    for label, class_vector in class_samples.items()}
                    label_mean_list = list(label_mean_dict.keys())
                    perm_list = list(permutations(label_mean_list, 2))
                    current_vector_label = low

                if class_thresholds is None:
                    class_thresholds = {i:1.0 for i in label_mean_list}

                sample_feature = sample[low:up].reshape(1, -1)
                for pred_label, gt_label in perm_list:
                    d_s1 = self.distance_function(sample_feature, label_mean_dict[pred_label])
                    d_s2 = self.distance_function(sample_feature, label_mean_dict[gt_label])
                    d_s2_s2 = self.distance_function(label_mean_dict[gt_label], label_mean_dict[gt_label])

                    sample_atoms.append(eval(f'{d_s1} {inequality} {class_thresholds[gt_label]} * {d_s2}'))
                    sample_atoms.append(eval(f'{d_s1} {inequality} {class_thresholds[gt_label]} * {d_s2_s2}'))
                    # if inequality == '>':
                    #     sample_atoms.append(d_s1 > class_thresholds[right]*d_s2)
                    #     sample_atoms.append(d_s1 > class_thresholds[right]*d_s2_s2)
                    # elif inequality == '>=':
                    #     sample_atoms.append(d_s1 >= class_thresholds[right]*d_s2)
                    #     sample_atoms.append(d_s1 >= class_thresholds[right]*d_s2_s2)
                    # elif inequality == '<':
                    #     sample_atoms.append(d_s1 < class_thresholds[right]*d_s2)
                    #     sample_atoms.append(d_s1 < class_thresholds[right]*d_s2_s2)
                    # elif inequality == '<=':
                    #     sample_atoms.append(d_s1 <= class_thresholds[right]*d_s2)
                    #     sample_atoms.append(d_s1 <= class_thresholds[right]*d_s2_s2)
                    # else:
                    #     sample_atoms.append(d_s1 == class_thresholds[right]*d_s2)
                    #     sample_atoms.append(d_s1 == class_thresholds[right]*d_s2_s2)
            final_list.append(sample_atoms)

        # Convert to array
        rule_array = np.array(final_list).reshape(-1, len(feature_vector_splits) * len(perm_list) * 2)

        # Create DataFrame
        cols = []
        if feature_names is not None:
            feature_names_vector_splits = [(k, i, j) for (i, j), k in zip(feature_vector_splits, feature_names)]
        else:
            feature_names_vector_splits = [(f'feat_{idx+1}', i, j) for idx, (i, j) in enumerate(feature_vector_splits)]

        for vector_type, low, up in feature_names_vector_splits: 
            current_vector_label = low
            for pred_label, gt_label in perm_list:
                cols.append(f'{vector_type}: {prefix}(s,{pred_label}) {inequality} {class_thresholds[gt_label]} * {prefix}(s,{gt_label})')
                cols.append(f'{vector_type}: {prefix}(s,{pred_label} {inequality} {class_thresholds[gt_label]} * {prefix}({gt_label},{gt_label})')

        rule_df = pd.DataFrame(rule_array, columns=cols)
        rule_df['label'] = self.label
        rule_df['pred_label'] = self.pred_label
        cols = list(rule_df.columns)
        cols = [cols[-2]] + [cols[-1]] + cols[:-2]
        rule_df = rule_df[cols]
        if label_dict is not None:
            rule_df['label'] = rule_df['label'].map(label_dict)
            rule_df['pred_label'] = rule_df['pred_label'].map(label_dict)
        return rule_df


        