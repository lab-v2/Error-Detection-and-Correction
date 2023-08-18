import numpy as np
import os
import pickle

class MaxRule:
    def __init__(self, speed_index_file = None):
        if os.path.isfile(speed_index_file):
            with open(speed_index_file, "rb") as f:
                self.max_speeds, self.max_speeds_index, self.dic_conf = pickle.load(f)
            self.hasfile = True
        else:
            self.max_speeds = [[0.255, 0.321, 0.358, 0.723, 0.752],
                              [0.228, 0.310, 0.321, 0.533, 0.690],
                              [0.219, 0.290, 0.321, 0.485, 0.662]]
            self.max_speeds_index = [[0, 2, 1, 3, 5],
                                     [0, 1, 2, 3, 5],
                                     [0, 1, 2, 3, 4]]
            self.dic_conf = {100:0, 98:1, 96:2}
            self.hasfile = False
        # chunk -> (10, 15), modes -> 1, condition -> ( >0.87 --> 4 )
        self.max_chunk_rules = [[(0, 101), 0, (0.752, 4)],
                          [(10, 15), 1, (0.87, 4)],
                          [(10, 15), 2, (0.17, 0)],
                          [(10, 15), 3, (0.90, 0)],
                          [(15, 20), 1, (0.89, 4)],
                          [(15, 20), 2, (0.20, 0)],
                          [(20, 25), 1, (0.89, 4)],
                          [(20, 25), 2, (0.20, 1)],
                          [(25, 30), 1, (0.90, 4)],
                          [(30, 35), 1, (0.91, 4)],
                          [(35, 40), 1, (0.91, 4)],
                          [(35, 40), 2, (0.20, 5)],
                          [(40, 45), 1, (0.94, 4)],
                          [(40, 45), 2, (0.23, 5)],
                          [(40, 45), 3, (0.94, 4)],
                          [(45, 50), 1, (0.94, 4)],
                          [(45, 50), 2, (0.24, 5)],
                          [(50, 55), 1, (0.92, 4)],
                          [(55, 60), 1, (0.92, 4)],
                          [(55, 60), 2, (0.22, 5)],
                          [(55, 60), 3, (0.88, 5)],
                          [(60, 65), 1, (0.94, 4)],
                          [(60, 65), 2, (0.22, 5)],
                          [(65, 70), 1, (0.94, 4)],
                          [(65, 70), 2, (0.19, 5)],
                          [(70, 75), 1, (0.93, 4)],
                          [(70, 75), 2, (0.20, 5)],
                          [(70, 75), 3, (0.64, 4)]]
    def rule_for_deterministic(self, max_values):
        expect = -1
        rule_results = []
        for count, crule in enumerate(self.max_chunk_rules):
            key = max_values[0] * 100
            mode = crule[1]
            threshold = crule[2][0]
            expection = crule[2][1]

            if key > crule[0][0] and key < crule[0][1]:
                if max_values[mode] > threshold:
                    rule_results.append([expection, count])
                    expect = expection
                else:
                    rule_results.append([-1, -1])
            else:
                rule_results.append([-1, -1])
        return expect, rule_results
    def rule_out_for_maxspeed(self, max_values, tk, conf):
        conf = self.dic_conf[conf]
        if self.hasfile:
            m0 = max_values[0]
            init = 0
            pnew = tk[init]
            maxchunk = 0
            rule_out = False
            for cii in self.max_speeds_index[conf]:
                ci = self.max_speeds[conf][cii]
                if m0 > ci and cii == pnew:
                    rule_out = True
                    break
            return rule_out
        else:
            print("File needed!")

    def rule_for_maxspeed(self, max_values, tk, conf):
        if self.hasfile:
            m0 = max_values[0]
            init = 0
            pnew = tk[init]
            maxchunk = 0
            for cii in self.max_speeds_index[conf]:
                ci = self.max_speeds[conf][cii]
                if m0 > ci:
                    maxchunk += 1
                else:
                    break
            #print(f"###pnew:{pnew}, maxchunk:{maxchunk}, tk:{tk}, m0:{m0}")
            while (pnew in self.max_speeds_index[conf][:maxchunk]):
                init += 1
                pnew = tk[init]
                if init > 4:
                    break
            return pnew
        else:
            m0 = max_values[0]
            init = 0
            pnew = tk[init]
            maxchunk = 0
            for ci in self.max_speeds[conf]:
                if m0 > ci:
                    maxchunk += 1
                else:
                    break
            #print(f"###pnew:{pnew}, maxchunk:{maxchunk}, tk:{tk}, m0:{m0}")
            while (pnew in self.max_speeds_index[conf][:maxchunk]):
                init += 1
                pnew = tk[init]
            return pnew

    def filter(self, max_values, pred, confidience = 100):
        if confidience not in self.dic_conf:
            print("Warning: confidence not suited, please check!")
            conf = self.dic_conf[100]
        else:
            conf = self.dic_conf[confidience]
        pnew = self.rule_for_maxspeed(max_values, pred, conf)
        return pnew

        if max_values[0] > self.max_speeds[conf][-1]:
            return pnew
        determin, rc = self.rule_for_deterministic(max_values)
        if determin != -1:
            return determin
        else:
            return pnew


