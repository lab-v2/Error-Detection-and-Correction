import numpy as np
import pandas as pd
import pickle
import os
import random
import time
from tqdm import tqdm

from itertools import permutations
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt

#classes = [0, 1, 2, 3, 4, 5]
classes = [0, 1]
obj = [ [] for _ in classes ]
#{'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'} # blue, green, red, cyan, magenta, yellow, black, and white
colors = ['b', 'g', 'r', 'y', 'k', 'c']
f1 = plt.figure()
for i in classes:
    filename = "/home/ubuntu/human_mvmt_0403/test{}.pkl".format(i)
    obj[i] = pd.read_pickle(filename)
    print(f"{i}", np.array(obj[i]).max(axis=(0,1)))
    dic_speed = {}
    for data in obj[i]:
        maxs = data.max(axis=0)
        key = int(maxs[0] * 100)
        if key not in dic_speed:
            dic_speed[key] = []
        dic_speed[key].append(maxs)
    keys = []
    values = []
    dic_speed = dict(sorted(dic_speed.items()))
    for key in dic_speed.keys():
        keys.append(key)
        values.append(np.array(dic_speed[key]).max(axis=(0)).tolist())
    #print(keys)
    #print(values)
    #plt.plot(keys, [v[1] for v in values],'-', color = colors[i])
    plt.plot(keys, [v[2] for v in values],'--', color = colors[i])
    #plt.plot(keys, [v[3] for v in values], '-.', color = colors[i])
plt.title(f"each class:{colors}")
plt.savefig(f"sub0-1-2.png")
        
        
'''
    print(obj[i][0].shape)
    #self.obj_mean.append(np.array(obj[i]).mean(axis = (0)))
    print(f"{i}", np.array(obj[i]).max(axis=(0,1)))
    #obj_length.append(len(obj[i]))

class Apoi:
    def __init__(self, train_files = None, num_classes = 6):
        self.classes = [i for i in range(num_classes)]
        self.num_classes = num_classes
    
        self.obj_mean = []
        self.predicate_diffs = []
        self.props = [] # each sentence probability + class probability
        self.combins2 = []
        self.combins3 = []
        for c in permutations(self.classes, 3):
            (c1,c2,c3) = c
            if c2 < c3:
                self.combins3.append(c)
        
        for c in combinations(self.classes, 2):
            self.combins2.append(c)
        self.class_thresholds = [1]
        self.channels = [0]
        #class_thresholds = [1.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        #channels = [0, 1, 2, 3]
    def compute_distance(self, array, base_array, c):
        [m,n] = np.array(base_array).shape
        base_diff = 1000
        for bi in range(m):
            diff = np.sum((array == base_array[bi]).astype(np.int))
            if diff == 0:
                return self.props[c][bi] * self.props[c][-1]
            #if base_diff > diff:
            #    base_diff = diff
            #if base_diff == 0:
            #    return 0
        #return base_diff
        return 0

    def get_closest_classes(self, array):
        if len(self.obj_mean) == 0 or len(self.predicate_diffs) == 0:
            self.create_features()

        pre_lists = self.distance_to_classify(array)
        pre_lists = np.asarray(pre_lists)
        results = [ [ ] for _ in range(len(pre_lists)) ]
        for count, val in enumerate(pre_lists):
            diss = np.zeros(self.num_classes)
            for c in self.classes:
                dis = self.compute_distance(val, self.predicate_diffs[c], c)
                diss[c] = dis
            results[count].append(diss) 
            #minvalue = np.min(diss)
            #if minvalue == 0:
            #    for countj , j in enumerate(self.classes):
            #        if diss[j] == minvalue:
            #            results[count][countj] = self.props.append(countj)
                
        return results

    def create_features(self, mean_file = None, predicate_file = None, prop_file = None):

        if mean_file and predicate_file and prop_file:
            if os.path.isfile(mean_file) and os.path.isfile(predicate_file) and os.path.isfile(prop_file):
                with open(mean_file, 'rb') as f:
                    self.obj_mean = pickle.load(f)
                with open(predicate_file, 'rb') as f:
                    self.predicate_diffs = pickle.load(f)
                with open(prop_file, 'rb') as f:
                    self.props = pickle.load(f)
                return
        obj = [ [] for _ in range(self.num_classes) ]
        obj_length = []
        # input array list(numpy) (-1,(40, 4))
        for i in self.classes:
            filename = "/home/ubuntu/human_mvmt_0403/test{}.pkl".format(i)
            obj[i] = pd.read_pickle(filename)
            self.obj_mean.append(np.array(obj[i]).mean(axis = (0)))
            print(f"{i}", np.array(obj[i]).max(axis=(0,1)))
            obj_length.append(len(obj[i]))
        total = sum(obj_length)

        if mean_file:
            with open(mean_file, "wb") as f:
                pickle.dump(self.obj_mean, f)

        predicate_lists = []
        #for i in classes:
        #    pre_lists = distance_to_classify_1(obj[i])
        #    predicate_lists.append(list(set(pre_lists)))
        
        for i in self.classes:
            obj[i] = np.array(obj[i]).transpose(0,2,1)
            pre_diffs = self.distance_to_classify(obj[i])
            results = Counter(pre_diffs)
            len_diffs = len(pre_diffs) * 1.0
            #predicate_lists.append(list(set(pre_lists)))
            self.predicate_diffs.append(list(set(pre_diffs)))
            #print(i, len(obj[i]), len(predicate_lists[i]), "###")
            prop = []
            for ele in self.predicate_diffs[i]:
                prop.append(results[ele] / len_diffs)
            prop.append(obj_length[i] / total)
            self.props.append(prop)

        
        for (c1,c2) in self.combins2:
            pre_coms = list(set(self.predicate_diffs[c1] + self.predicate_diffs[c2]))
            len_c1 = len(self.predicate_diffs[c1])
            len_c2 = len(self.predicate_diffs[c2])
            print(f"{c1} length: {len_c1},\
                  {c2} length: {len_c2},\
                  {c1} + {c2} length: {len_c1 + len_c2} {len(pre_coms)} {max(len_c1, len_c2)}")
        
        #filename = "xbw/feature_lists.pkl"
        self.predicate_diffs = np.asarray(self.predicate_diffs)
        if predicate_file:
            with open(predicate_file, "wb") as f:
                pickle.dump(self.predicate_diffs, f)
        if prop_file:
            with open(prop_file, "wb") as f:
                pickle.dump(self.props, f)



    def cosine_similarity(self, A, B):
        dot_product = A.dot(B)
        norm_of_A = np.linalg.norm(A)
        norm_of_B = np.linalg.norm(B)
        denominator = norm_of_A * norm_of_B
        if denominator == 0:
            return 0
        cos = dot_product / denominator
        return cos

    def distance_to_classify(self, diff_data):
        predicate_list_all = []
        #predicate_list_diff = []
        length = diff_data.shape[0]
        for i in (range(length)):
            predicate_list = []
            #predicate_list.append(cla)
            
            for ratio in self.class_thresholds:
                for j in self.channels: # 4 channels
                    distance = np.zeros(self.num_classes)
                    for k in self.classes:
                        #distance[k] =np.linalg.norm(diff_data[i,j,:] - obj_mean[k][:,j]) 
                        distance[k] =self.cosine_similarity(diff_data[i,j,:], self.obj_mean[k][:,j]) 
                    #distance[cla] *= ratio
                    #distance_to_train_true *= ratio
                    for (c1, c2, c3) in self.combins3:
                        lab1 = 10000 + c1 * 1000 + c2 * 100 + c3 * 10 + c1 + ratio * 100000
                        lab2 = 10000 + c1 * 1000 + c2 * 100 + c3 * 10 + c2 + ratio * 100000
                        predicate_list.append(lab1 if distance[c1] >= (distance[c2] + distance[c3])/2.0 else lab2)
        
            predicate_list_all.append(tuple(predicate_list))
            #predicate_list_diff.append(tuple(predicate_list[1:]))
            #predicate_list_true.append(tuple(predicate_list))
        return predicate_list_all

#distance_to_false(f5_i, false_train_data_all)
#distance_to_true(t5_i, true_train_data_all)

#from efficient_apriori import apriori
#for i in classes:
#    filename = "xbw/rule_{}.csv".format(i)
#    itemsets, rules = apriori(predicate_lists[i], output_transaction_ids=True)
#    df_rule = pd.DataFrame(rules)
#    df_rule.to_csv(filename)
#
#def distance_to_classify_1(diff_data):
#    predicate_list_all = []
#    predicate_list_diff = []
#    length = len(diff_data)
#    diff_data = np.array(diff_data).transpose(0,2,1)
#    for i in range(length):
#        predicate_list = []
#        
#        for ratio in class_thresholds:
#            for j in channels: # 4 channels
#                distance = np.zeros(self.num_classes)
#                for k in classes:
#                    distance[k] =np.linalg.norm(diff_data[i,j,:] - obj_mean[k][:,j]) 
#                #distance[cla] *= ratio
#                #distance_to_train_true *= ratio
#                for (c1,c2) in combins2:
#                    lab1 = 1000 + c1 * 100 + c2 * 10 + c1 + ratio * 100000
#                    lab2 = 1000 + c1 * 100 + c2 * 10 + c2 + ratio * 100000
#                    predicate_list.append(lab1 if distance[c1] >= distance[c2] else lab2)
#    
#        predicate_list_all.append(tuple(predicate_list))
#        #predicate_list_true.append(tuple(predicate_list))
#    return predicate_list_all
#
#predicate_list_false = distance_to_classify(f5_i, false_train_data_all, 0)
#predicate_list_true = distance_to_classify(t5_i, true_train_data_all, 1)
#
#predicate_list_false = list(set(predicate_list_false))
#predicate_list_true = list(set(predicate_list_true))
#predicate_list_diff = list(set(predicate_list_diff))
#
#print("predicate_list_true:",len(predicate_list_true))
#print("predicate_list_false:",len(predicate_list_false))
#print("predicate_list_all:",len(predicate_list_diff))
##itemsets, rules = apriori(predicate_list_all, min_support=0.179, min_confidence=0.98)
##itemsets, rules = apriori(predicate_list_all, output_transaction_ids=True)
##print(type(itemsets))
##print(itemsets)
#df_rule = pd.DataFrame(rules)
#df = pd.DataFrame(itemsets).T
#df1 = pd.DataFrame(predicate_list_all, columns=cols)
##print(df1.head)
#
##df = (df.T)
#
##print (df)
#df_rule.to_csv('./xbw/rule_179_98.csv')
#df.to_csv('./xbw/itemset_179_98.csv')
##df1.to_excel('./qzy_data/dict1.xlsx')
#df1.to_csv('./xbw/dict1_179_98.csv')
'''
