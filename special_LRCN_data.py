#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import numpy as np
import pickle
import os
from scipy.signal import savgol_filter
import random
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

print(os.getcwd())
filename = './25_sequence_Revised_InstanceCreation+NoJerkOutlier+Smoothing.pickle'
# Each of the following variables contain multiple lists, where each list belongs to a user
with open(filename, 'rb') as f:
    Total_RelativeDistance, Total_Speed, Total_Acceleration, Total_Jerk, Total_BearingRate, Total_Label,\
    Total_InstanceNumber, Total_Instance_InSequence, Total_Delta_Time, Total_Velocity_Change = pickle.load(f, encoding='latin1')

# Create the data in the Keras form
# Threshold: Is the max of number of GPS point in an instance

Indexs = [i for i in range(len(Total_InstanceNumber))]
random.seed(42)
random.shuffle(Indexs)

ctmax = 0
ctmin = 1000
spmax = 0
spmin = 10000
ACmax = 0
ACmin = 10000
Jmax = 0
Jmin = 10000
BRmax = 0
BRmin = 1000
for index in Indexs:
    RD = Total_RelativeDistance[index][0]
    SP = Total_Speed[index][0]
    AC = Total_Acceleration[index][0]
    J = Total_Jerk[index][0]
    BR = Total_BearingRate[index][0]
    LA = Total_Label[index][0]
    IN = Total_InstanceNumber[index][0]
    if IN < 1:
        continue
    if max(SP) > spmax:
        spmax = max(SP)
    if min(SP) < spmin:
        spmin = min(SP)

    if max(AC) > ACmax:
        ACmax = max(AC)
    if min(AC) < ACmin:
        ACmin = min(AC)

    if max(J) > Jmax:
        Jmax = max(J)
    if min(J) < Jmin:
        Jmin = min(J)

    if max(BR) > BRmax:
        BRmax = max(BR)
    if min(BR) < BRmin:
        BRmin = min(BR)
print(spmax,spmin)
print(ACmax,ACmin)
print(Jmax,Jmin)
print(BRmax,BRmin)
# Method 0: 40 * 5 no overlap
# Method 1: 40 * 5 sample overlap 
# Method 2: 40 * 5 segment overlap 

Threshold = 200
Number_Sequence = 5
each_sequence = 40
features = 4 # sp, ac, j, br
in_classes = set([0, 2, 4])
def methodforsample():
    All_sequences = []
    for index in Indexs:
        len_labels = len(Total_InstanceNumber[index])
        each_index = []
        each_label = []
        for lb in range(len_labels):
            RD = Total_RelativeDistance[index][lb]
            SP = Total_Speed[index][lb]
            AC = Total_Acceleration[index][lb]
            J = Total_Jerk[index][lb]
            BR = Total_BearingRate[index][lb]
            LA = Total_Label[index][lb]
            IN = Total_InstanceNumber[index][lb]
            if IN >= each_sequence and LA in in_classes:
                tempI = [[] for _ in range(features)]
                for i in range(IN):
                    tempI[0].append(SP[i])
                    tempI[1].append(AC[i])
                    tempI[2].append(J[i])
                    tempI[3].append(BR[i])
                    if len(tempI[0]) == each_sequence:
                        each_index.append(tempI)
                        each_label.append(LA)
                        tempI = [[] for _ in range(features)]
        All_sequences.append([each_index, each_label])
    return All_sequences

def methodforsegment(offset = 10):
    All_sequences = []
    for index in Indexs:
        len_labels = len(Total_InstanceNumber[index])
        each_index = []
        each_label = []
        for lb in range(len_labels):
            RD = Total_RelativeDistance[index][lb]
            SP = Total_Speed[index][lb]
            AC = Total_Acceleration[index][lb]
            J = Total_Jerk[index][lb]
            BR = Total_BearingRate[index][lb]
            LA = Total_Label[index][lb]
            IN = Total_InstanceNumber[index][lb]
            if IN >= each_sequence + offset:
                tempI = [[] for _ in range(features)]
                for i in range(IN):
                    tempI[0].append(SP[i])
                    tempI[1].append(AC[i])
                    tempI[2].append(J[i])
                    tempI[3].append(BR[i])
                    if len(tempI[0]) == each_sequence:
                        each_index.append(tempI)
                        each_label.append(LA)
                        tempI = [[] for _ in range(features)]
                All_sequences.append([each_index, each_label])
    return All_sequences

def method0():
    All_sequences = methodforsample()
    Samples = []
    for each_features, each_label in All_sequences:
        length = len(each_features)
        for i in range(0, length - 5, 5):
            Samples.append([each_features[i:i+5],each_label[i:i+5]])
    return Samples

def method1():
    All_sequences = methodforsample()
    Samples = []
    for each_features, each_label in All_sequences:
        length = len(each_features)
        for i in range(0, length - 5):
            Samples.append([each_features[i:i+5],each_label[i:i+5]])
    return Samples

def method2(offset):
    All_sequences = methodforsegment(offset)
    Samples = []
    for each_features, each_label in All_sequences:
        length = len(each_features)
        for i in range(0, length - 5, 5):
            Samples.append([each_features[i:i+5],each_label[i:i+5]])
    return Samples

def generateRandomTrainTest(Samples):
    tempTotalInput = [a[0] for a in Samples]
    tempFinalLabel = [a[1] for a in Samples]
    tempTotalInput=np.array(tempTotalInput, dtype=float)
    tempFinalLabel=np.array(tempFinalLabel, dtype=int)
    print(tempTotalInput.shape)
    print(tempFinalLabel.shape)
    #(19359, 5, 4, 40)
    tempLabels = np.reshape(tempFinalLabel,(-1,))
    print(Counter(tempLabels))
    
    scaler = MinMaxScaler()
    [a1, b1, c1, d1] = tempTotalInput.shape
    tempTotalInput = np.transpose(tempTotalInput, (0, 1, 3, 2))
    tempTotalInput = np.reshape(tempTotalInput, (-1,c1))
    tempTotalInput = scaler.fit_transform(tempTotalInput)
    tempTotalInput = np.reshape(tempTotalInput, (a1, b1, d1, c1))
    tempTotalInput = np.transpose(tempTotalInput, (0, 1, 3, 2))
    print(scaler.data_max_)
    print(scaler.data_min_)
    print("input[:5]", tempTotalInput[0,0,:,:5])
    
    X_train, X_test, y_train, y_test = train_test_split(tempTotalInput, tempFinalLabel, test_size=0.3, random_state=42)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.33, random_state=42)
    print(f"train length:{len(X_train)}, valid length:{len(X_valid)}, test length:{len(X_test)}")

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def generateSequentialTrainTest(Samples):
    train_ratio = 0.7
    valid_ratio = 0.8
    tempTotalInput = [a[0] for a in Samples]
    tempFinalLabel = [a[1] for a in Samples]
    tempTotalInput=np.array(tempTotalInput, dtype=float)
    tempFinalLabel=np.array(tempFinalLabel, dtype=int)
    print(tempTotalInput.shape)
    #(19359, 5, 4, 40)
    tempLabels = np.reshape(tempFinalLabel,(-1,))
    print(Counter(tempLabels))
    
    scaler = MinMaxScaler()
    [a1, b1, c1, d1] = tempTotalInput.shape
    tempTotalInput = np.transpose(tempTotalInput, (0, 1, 3, 2))
    tempTotalInput = np.reshape(tempTotalInput, (-1,c1))
    tempTotalInput = scaler.fit_transform(tempTotalInput)
    tempTotalInput = np.reshape(tempTotalInput, (a1, b1, d1, c1))
    tempTotalInput = np.transpose(tempTotalInput, (0, 1, 3, 2))
    
    train_length = int(a1 * train_ratio)
    valid_length = int(a1 * valid_ratio)
    X_train = tempTotalInput[:train_length]
    y_train = tempFinalLabel[:train_length]
    X_valid = tempTotalInput[train_length:valid_length]
    y_valid = tempFinalLabel[train_length:valid_length]
    X_test = tempTotalInput[valid_length:]
    y_test = tempFinalLabel[valid_length:]
    print(f"train length:{train_length}, valid length:{valid_length - train_length}, test length:{a1 - valid_length}")
    return X_train, y_train, X_valid, y_valid, X_test, y_test

overlap_types=['no_overlap', 'sample_overlap', 'segment_overlap']
sample_types=['random', 'sequential']
num_overlap = int(sys.argv[1])
num_sample = int(sys.argv[2])
offset = int(sys.argv[3])
if num_overlap > 2:
    print("Overlap type parameter error! num_overlap should between [0,2], please check!")
if num_sample > 1:
    print("Sample type parameter error! num_sample should between [0,1], please check!")
if num_overlap == 0:
    Samples = method0()
if num_overlap == 1:
    Samples = method1()
if num_overlap == 2:
    Samples = method2(offset)
if num_sample == 0:
    [X_train, y_train, X_valid, y_valid, X_test, y_test] = generateRandomTrainTest(Samples)
if num_sample == 1:
    [X_train, y_train, X_valid, y_valid, X_test, y_test] = generateSequentialTrainTest(Samples)

with open(f'./special_train_valid_test_{overlap_types[num_overlap]}_{sample_types[num_sample]}.pickle', 'wb') as f:
   pickle.dump([X_train, y_train, X_valid, y_valid, X_test, y_test], f)
