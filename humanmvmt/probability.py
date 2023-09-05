import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from scipy.stats import entropy
import torch

class Probability:
    def __init__(self, dataloader = None, saved_prob_path = None, saved_means_path = None):
        self.dataloader = dataloader
        self.saved_prob_path = saved_prob_path
        self.saved_means_path = saved_means_path

    #def mutual_information(self, x, y, bins=10):
    #    # Create histograms of x and y
    #    hist_x, edges_x = np.histogram(x, bins=bins, density=True)
    #    hist_y, edges_y = np.histogram(y, bins=bins, density=True)
    #
    #    # Compute joint histogram
    #    hist_xy, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    #
    #    # Compute PDFs
    #    pdf_x = hist_x * np.diff(edges_x)
    #    pdf_y = hist_y * np.diff(edges_y)
    #    pdf_xy = hist_xy * np.outer(np.diff(edges_x), np.diff(edges_y))
    #
    #    # Compute entropies
    #    h_x = -np.sum(pdf_x * np.log2(pdf_x + 1e-12))
    #    h_y = -np.sum(pdf_y * np.log2(pdf_y + 1e-12))
    #    h_xy = -np.sum(pdf_xy * np.log2(pdf_xy + 1e-12))
    #
    #    # Compute mutual information
    #    mi = h_x + h_y - h_xy
    #
    #    return mi

    #def cosine_similarity(self, A: np.ndarray, B: np.ndarray) -> float:
    #    return entropy(A,qk = B)
    def cosine_similarity(self, A: np.ndarray, B: np.ndarray) -> float:
        '''Calculates the cosine similarity between two arrays
    
        Args:
           A: a numpy array which corresponds to a word vector
           B: A numpy array which corresponds to a word vector
        Return:
           cos: numerical number representing the cosine similarity between A and B.
        '''
        dot_product = A.dot(B)
        norm_of_A = np.linalg.norm(A)
        norm_of_B = np.linalg.norm(B)
        denominator = norm_of_A * norm_of_B
        if denominator == 0:
            return 0
        cos = dot_product/denominator
        return cos
    def compute_probability(self,value):
        '''
        input shape (40, 4)
        return:
            argmin(40,0) -> (1,)
        or 
        input shape (batch, 40 , 4)
        return:
            argmin(batch,40, 0) -> (batch,)
        or 
        input shape(batch, 5, 4, 40)
        return:
            argmin(batch,5,0,40) -> (batch, 5,)
        '''
        results = np.zeros(6)
        if type(value) == torch.Tensor:
            value = value.cpu().detach().numpy()
        ndim = value.ndim
        if ndim == 2:
            fe = []
            for k in self.all_keys:
                ms = self.means[k]
                for kk in range(n_dimension):
                    cos = self.cosine_similarity(value[:,kk],ms[:,kk])
                    #cos = self.mutual_information(value[:,kk], ms[:,kk], 20)
                    fe.append(cos)
            if sum(fe) == 0:
                results[-1] = 10000
                return results

            a = np.reshape(np.array(fe),(-1,n_dimension))
            a = np.argmin(a, axis = 0)
            for i in range(a.shape[0]):
                results[:-1] += self.probability[i,:,a[i]]
                
            return results 
        elif ndim == 3:
            features = []
            for v in value:
                fe = []
                for k in self.all_keys:
                    ms = self.means[k]
                    cos = self.cosine_similarity(v[:,0],ms[:,0])
                    #cos = self.mutual_information(v[:,0], ms[:,0], 20)
                    fe.append(cos)
                if sum(fe) == 0:
                    features.append(np.zeros(6))
                    continue
                a = np.argmin(np.array(fe))
                p = self.probability[:,a]
                features.append(p)
            return np.array(features)
        elif ndim == 4:
            [n_batch, n_size, n_dimension, n_length] = value.shape
            features = []
            for batch in range(n_batch):
                features.append([])
                for s in range(n_size):
                    features[batch].append([])
                    results = np.zeros((4,6))

                    fe = []
                    for k in self.all_keys:
                        ms = self.means[k]
                        for kk in range(n_dimension):
                            cos = self.cosine_similarity(value[batch, s,kk,:],ms[:,kk])
                            #cos = self.mutual_information(value[batch, s, kk, :], ms[:,kk], 20)
                            fe.append(cos)
                    #if sum(fe) == 0:
                    #    results[:,-1] = 1
                    #    features[batch][s].append(results)
                    #    continue

                    a = np.reshape(np.array(fe),(-1,n_dimension))
                    a = np.argmax(a, axis = 0)
                    for i in range(a.shape[0]):
                        results[i] = self.probability[i,:,a[i]]
                    features[batch][s].append(results)
            features = np.squeeze(np.array(features), axis = 2)
            return features
                    
            


    # 先做统计，求均值方差
    def set_probability(self):
        if self.saved_prob_path is not None and os.path.isfile(self.saved_prob_path):
            print(f"*** Probability file exists, read probability from the {self.saved_prob_path} ***")
            self.probability = np.load(self.saved_prob_path)
            print("probability is:",self.probability)

        if self.saved_means_path is not None and os.path.isfile(self.saved_means_path):
            print(f"*** means file exists, read means from the {self.saved_means_path} ***")
            with open(self.saved_means_path, 'rb') as f:
                self.means = pickle.load(f)
            #print("means is:",self.means)
            all_keys = list(self.means.keys())
            all_keys.sort()
            self.all_keys = all_keys
            return
    
        dicts = {}
        for (datas, labels) in tqdm(self.dataloader):
            all_datas = datas.numpy()
            all_labels = labels.numpy()
    
            all_datas = np.transpose(all_datas, (0,3,2,1))
            [n_size, n_length, n_dimension, n_row] = all_datas.shape
            for i in (range(n_size)):
                for j in range(n_row):
                    if all_labels[i,j] not in dicts:
                        dicts[all_labels[i,j]] = []
                    dicts[all_labels[i,j]].append(all_datas[i,:,:,j])
    
        #dicts = dict.fromkeys(keys, [])

        #means = {} #dict.fromkeys(keys, [])
        #
        #for key, values in dicts.items():
        #    ms = np.mean(np.array(values), axis=0)
        #    means[key] = ms
        #if self.saved_means_path :
        #    with open(self.saved_means_path, 'wb') as f:
        #        pickle.dump(means, f)

        #self.means = means 

        all_keys = list(self.means.keys())
        all_keys.sort()
        self.all_keys = all_keys
        print("keys:", all_keys)
        
        features = {}
        for key, values in dicts.items():
            features[key] = []
            for value in tqdm(values, desc = "Processing key:%s"%(key)):
                fe = []
                for k in all_keys:
                    ms = self.means[k]
                    for k in range(n_dimension):
                        cos = self.cosine_similarity(value[:,k],ms[:,k])
                        #cos = self.mutual_information(value[:,k], ms[:,k], 20)
                        fe.append(cos)
                features[key].append(fe)
                
        with open("Models/cosine_similarity.pkl", 'wb') as f:
            pickle.dump(features, f)
        
        probability = np.zeros((n_dimension, len(all_keys),len(all_keys)))
        with open("Models/cosine_similarity.pkl", 'rb') as f:
            features = pickle.load(f)
            for k in all_keys:
                counts = [{} for i in range(n_dimension)]
                feature = features[k]
                for f in feature:
                    a = np.reshape(np.array(f),(-1,n_dimension))
                    a = np.argmax(a, axis = 0)
                    for i in range(a.shape[0]):
                        if a[i] not in counts[i]:
                            counts[i][a[i]] = 0
                        counts[i][a[i]] += 1
                length = len(feature)
                w1 = int(k)
                for ii in range(n_dimension):
                    for kk, vv in counts[ii].items():
                        probability[ii][w1][kk] = vv/length
                #for c, count in enumerate(counts):
                #    print(f"#####{k}, {c-1}#####")
                #    for kk, vv in count.items():
                #        print( kk, vv/length)
            print(probability[0])
            print(probability[1])
            print(probability[2])
            print(probability[3])
        for i in range(n_dimension):
            probability[i] = probability[i] / np.sum(probability[i], axis = 0)
        probability = np.nan_to_num(probability)
        self.probability = probability
        if self.saved_prob_path is not None:
            with open(self.saved_prob_path, 'wb') as f:
                np.save(f, probability)
        print("probability:\n",probability)
        return probability

#        features = []
#        features.append(all_labels[i,j])
#        for k in range(n_dimension):
#            ts = pd.Series(all_datas[i,:,k,j]).ewm( alpha =0.5).mean().mean()
#            features.append(ts)
#        all_features.append(features)

