import numpy as np
import numpy
import torch
# from speed_rule import MaxRule
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report
import matplotlib.pyplot as plt
import sys

base_path0=sys.argv[1]  # neural network predict and true label
base_path1=sys.argv[2]  # rules
true_file = base_path0 + "test_true.npy"
pred_file = base_path0 + "test_pred.npy"
tout_file = base_path0 + 'test_out.npy'

true_data = np.load(true_file,allow_pickle=True)
pred_data = np.load(pred_file,allow_pickle=True)
tout_data = np.load(tout_file,allow_pickle=True)
#true_data = np.asarray([i.numpy() for i in true_data])
cla4_data = np.load(base_path1 + "test_out_cla4.npy",allow_pickle=True)
cla3_data = np.load(base_path1 + "test_out_cla3.npy",allow_pickle=True)
cla2_data = np.load(base_path1 + "test_out_cla2.npy",allow_pickle=True)
cla1_data = np.load(base_path1 + "test_out_cla1.npy",allow_pickle=True)
cla0_data = np.load(base_path1 + "test_out_cla0.npy",allow_pickle=True)

#logsofmax = torch.nn.LogSoftmax(dim = 1)
#lsm = logsofmax(torch.from_numpy(orig_data[0]))
#print(lsm)
labels = set(true_data.flatten())
len_labels = len(labels)

#print(labels)
#print(true_data.shape, true_data[0])
#print(pred_data.shape, pred_data[0])
#
print("tout_data:\n",tout_data.shape, tout_data[0])
print("cla0_data:\n",cla0_data.shape, cla0_data[0])
#
#print(appr_data.shape, appr_data[0])
#print(apla_data.shape, apla_data[0])

n_classes = 5
count = 0
c1 = 0
maxRules = MaxRule("confidents_obj.pkl")
charts = []
print(maxRules.dic_conf)
#confidences = sorted(maxRules.dic_conf.keys())
confidences = [i * 0.01 for i in range(1, 100, 10)]
pr_curve = []
print(f"confidences:{confidences}")
cla_datas = [cla0_data, cla1_data, cla2_data, cla3_data, cla4_data]   # neural network binary result
high_scores = [0.95, 0.98]  # >0.95 is one rule, >0.98 is another rule, in total 4*5
low_scores = [0.05, 0.02]
#high_scores = [0.8]
#low_scores = [0.2]

def rules1(i):
    rule_scores = []
    for cls in cla_datas:
        for score in high_scores:
            if cls[i] > score:
                rule_scores.append(1)
            else:
                rule_scores.append(0)
        for score in low_scores:
            if cls[i] < score:
                rule_scores.append(1)
            else:
                rule_scores.append(0)
    return rule_scores
def rules2(i, j):
    rule_scores = []
    for cls in cla_datas:
        for score in high_scores:
            if cls[i,j] > score:
                rule_scores.append(1)
            else:
                rule_scores.append(0)
        for score in low_scores:
            if cls[i,j] < score:
                rule_scores.append(1)
            else:
                rule_scores.append(0)
    return rule_scores

if true_data.ndim == 2:
    [m,n] = true_data.shape
    print(f"m:{m}, n:{n}, {m*n}")

    for i in range(m):
        #print(true_data[i])
        for j in range(n):
            index = i * n + j
            tout = tout_data[i,:,j]
            #tk = torch.topk(torch.from_numpy(tout), 2)
            tk = np.argsort(tout)[::-1]
            #tke = torch.exp(torch.from_numpy(tout))
    
            pnew = tk[0]
            tmp_charts = []
            tmp_charts.extend([tk[0], true_data[i,j]])
            tmp_charts += rules2(i, j)
            charts.append(tmp_charts) 
else:
    m = true_data.shape[0]
    for i in range(m):
        tmp_charts = []
        tmp_charts.extend([pred_data[i], true_data[i]])
        tmp_charts += rules1(i)
        charts.append(tmp_charts)





def get_scores(y_true, y_pred):
    try:
        y_actual = y_true
        y_hat = y_pred
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
               TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
               FP += 1
            if y_actual[i]==y_hat[i]==0:
               TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
               FN += 1
        print(f"TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}")
        
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return [pre, rec, f1]
    except:
        pre = accuracy_score(y_true, y_pred)
        f1 =         f1_score(y_true, y_pred, average='macro')
        f1micro =         f1_score(y_true, y_pred, average='micro')
        return [pre, f1, f1micro]


def generate_chart(charts):
    all_charts = [[] for _ in range(n_classes)]
    for data in charts:
        for count, jj in enumerate(all_charts):
            # pred, corr, tp, fp, cond1, cond2 ... condn
            each_items = []
            for d in data[:2]:
                if d == count:
                    each_items.append(1)
                else:
                    each_items.append(0)

            if each_items[0] == 1 and each_items[1] == 1:
                each_items.append(1)
            else:
                each_items.append(0)
            if each_items[0] == 1 and each_items[1] == 0:
                each_items.append(1)
            else:
                each_items.append(0)
            each_items.extend(data[2:])
            jj.append(each_items)
    return all_charts

def DetUSMPosRuleSelect(i, all_charts):
    count = i
    chart = all_charts[i]
    chart = np.array(chart)
    rule_indexs = [i for i in range(4, len(chart[0]))]
    each_sum = np.sum(chart, axis = 0)
    tpi = each_sum[2]
    fpi = each_sum[3]
    pi = tpi * 1.0 /(tpi + fpi)

    pb_scores = []
    for ri in rule_indexs:
        posi = np.sum(chart[:,1] * chart[:,ri], axis = 0)
        bodyi = np.sum(chart[:,ri], axis = 0)
        score = posi * 1.0 / bodyi
        if score > pi:
            pb_scores.append((score, ri))
    pb_scores = sorted(pb_scores)
    cci = []
    ccn = pb_scores
    for (score, ri) in pb_scores:

        cii = 0
        ciij = 0
        for (cs, ci) in cci:
            cii = cii | chart[:,ci]
        POScci = np.sum(cii * chart[:, 1], axis = 0)
        BODcci = np.sum(cii, axis = 0)
        POSccij = np.sum((cii | chart[:,ri]) * chart[:, 1], axis = 0)
        BODccij = np.sum((cii | chart[:,ri]), axis = 0)

        cni = 0
        cnij = 0
        for (cs, ci) in ccn:
            cni = (cni | chart[:,ci])
            if ci == ri:
                continue
            cnij = (cnij | chart[:, ci])
        POScni = np.sum(cni * chart[:, 1], axis = 0)
        BODcni = np.sum(cni, axis = 0)
        POScnij = np.sum(cnij * chart[:, 1], axis = 0)
        BODcnij = np.sum(cnij, axis = 0)

        a = POSccij * 1.0 / (BODccij + 0.001) - POScci * 1.0 / (BODcci + 0.001)
        b = POScnij * 1.0 / (BODcnij + 0.001) - POScni * 1.0 / (BODcni + 0.001)
        if a >= b:
            cci.append((score, ri))
        else:
            ccn.remove((score, ri))

    cii = 0
    for (cs, ci) in cci:
        cii = cii | chart[:,ci]
    POScci = np.sum(cii * chart[:, 1], axis = 0)
    BODcci = np.sum(cii, axis = 0)
    new_pre = POScci * 1.0 / (BODcci + 0.001)
    if new_pre < pi:
        cci = []
    cci = [c[1] for c in cci]
    print(f"class{count}, cci:{cci}, new_pre:{new_pre}, pre:{pi}")
    return cci


import itertools
def GreedyNegRuleSelect(i, epsilon, all_charts):
    count = i
    chart = all_charts[i]
    chart = np.array(chart)
    rule_indexs = [i for i in range(4, len(chart[0]))]
    len_rules = len(rule_indexs)
    each_sum = np.sum(chart, axis = 0)
    tpi = each_sum[2]
    fpi = each_sum[3]
    pi = tpi * 1.0 /(tpi + fpi)
    ri = tpi * 1.0 / each_sum[1]
    ni = each_sum[0]
    quantity = epsilon * ni * pi / ri
    print(f"class{count}, quantity:{quantity}")

    best_combins = []
    NCi = []
    NCn = []
    for rule in rule_indexs:
        negi_score = np.sum(chart[:,2] * chart[:,rule])
        if negi_score < quantity:
            NCn.append(rule)

    while(NCn):
        best_score = -1
        best_index = -1
        for c in NCn:
            tem_cond = 0
            for cc in NCi:
                tem_cond |= chart[:,cc]
            tem_cond |= chart[:,c]
            posi_score = np.sum(chart[:,3] * tem_cond)
            if best_score < posi_score:
                best_score = posi_score
                best_index = c
        NCi.append(best_index)
        NCn.remove(best_index)
        tem_cond = 0
        for cc in NCi:
            tem_cond |= chart[:,cc]
        tmp_NCn = []
        for c in NCn:
            tem = tem_cond | chart[:,c]
            negi_score = np.sum(chart[:,2] * tem)
            if negi_score < quantity:
                tmp_NCn.append(c)
        NCn = tmp_NCn
    print(f"class:{count}, NCi:{NCi}")
    return NCi

    for r in range(1,len_rules + 1):
        combinations = list(itertools.combinations(rule_indexs, r))
        max_score = [0, 0, 0]
        max_combi = tuple()
        for cond in combinations:
            tmp_cond = 0
            for c in cond:
                tmp_cond |= chart[:,c]
            negi = chart[:,2] * tmp_cond 
            negi_score = np.sum(negi)
            if negi_score < quantity:
                posi = chart[:,3] * tmp_cond
                posi_score = np.sum(posi)
                if posi_score - negi_score > max_score[0]:
                    max_score[0] = posi_score - negi_score
                    max_score[1] = negi_score
                    max_score[2] = posi_score
                    max_combi = cond
        print(f"class{count}, r:{r}, max_score:{max_score[0]}, negi:{max_score[1]}, posi:{max_score[2]}, max_combi:{max_combi}")
        if max_combi:
            best_combins.append(max_combi)
    return best_combins

def GreedyNegRules(all_charts):
    epsilon = 0.01
    for count, chart in enumerate(all_charts):
        GreedyNegRuleSelect(count, epsilon, all_charts)
            
def DetUSMPosRules(all_charts):
    for count, chart in enumerate(all_charts):
        DetUSMPosRuleSelect(count, all_charts)

def ruleForPNCorrection(all_charts, epsilon):
    results = []
    total_results = np.copy(pred_data)
    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        negi_count = 0
        posi_count = 0

        predict_result = np.copy(chart[:,0])
        CCi = []
        CCi = DetUSMPosRuleSelect(count, all_charts)
        tem_cond = 0
        for cc in CCi:
            tem_cond |= chart[:,cc]
        if np.sum(tem_cond) > 0:
            for ct,cv in enumerate(chart):
                if tem_cond[ct]:
                    if not predict_result[ct]:
                        posi_count += 1
                        predict_result[ct] = 1
                        total_results[ct] = count

        NCi = GreedyNegRuleSelect(count, epsilon, all_charts)

        tem_cond = 0
        for cc in NCi:
            tem_cond |= chart[:,cc]
        if np.sum(tem_cond) > 0:
            for ct,cv in enumerate(chart):
                if tem_cond[ct] and predict_result[ct]:
                    negi_count += 1
                    predict_result[ct] = 0


        scores_cor = get_scores(chart[:,1], predict_result)
        results.extend(scores_cor + [ negi_count, posi_count, len(NCi), len(CCi) ])
    results.extend(get_scores(true_data, total_results))
    return results

def ruleForNegativeCorrection(all_charts, epsilon):   #how to use
    results = []
    total_results = np.copy(pred_data)
    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        NCi = GreedyNegRuleSelect(count, epsilon, all_charts)
        negi_count = 0
        posi_count = 0

        predict_result = np.copy(chart[:,0])
        tem_cond = 0
        for cc in NCi:
            tem_cond |= chart[:,cc]
        if np.sum(tem_cond) > 0:
            for ct,cv in enumerate(chart):
                if tem_cond[ct] and predict_result[ct]:
                    negi_count += 1
                    predict_result[ct] = 0

        CCi = []
        scores_cor = get_scores(chart[:,1], predict_result)
        results.extend(scores_cor + [ negi_count, posi_count, len(NCi), len(CCi) ])
    results.extend(get_scores(true_data, total_results))
    return results

def ruleForNPCorrection(all_charts, epsilon):
    results = []
    total_results = np.copy(pred_data)
    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        NCi = GreedyNegRuleSelect(count, epsilon, all_charts)
        negi_count = 0
        posi_count = 0

        predict_result = np.copy(chart[:,0])
        tem_cond = 0
        for cc in NCi:
            tem_cond |= chart[:,cc]
        if np.sum(tem_cond) > 0:
            for ct,cv in enumerate(chart):
                if tem_cond[ct] and predict_result[ct]:
                    negi_count += 1
                    predict_result[ct] = 0

        CCi = []
        CCi = DetUSMPosRuleSelect(count, all_charts)
        tem_cond = 0
        rec_true = []
        rec_pred = []
        for cc in CCi:
            tem_cond |= chart[:,cc]
        if np.sum(tem_cond) > 0:
            for ct,cv in enumerate(chart):
                if tem_cond[ct]:
                    if not predict_result[ct]:
                        posi_count += 1
                        predict_result[ct] = 1
                        total_results[ct] = count
                else:
                    rec_true.append(cv[1])
                    rec_pred.append(cv[0])

        scores_cor = get_scores(chart[:,1], predict_result)
        results.extend(scores_cor + [ negi_count, posi_count, len(NCi), len(CCi) ])
    results.extend(get_scores(true_data, total_results))
    return results


def PosNegRuleLearn(all_charts):
    epsilon = 0.01
    #pi = [[] for _ in range(6)]
    #CCall = [[] for _ in range(6)]
    pi = []
    CCall = []
    CCall_set = []
    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        NCi = GreedyNegRuleSelect(count, epsilon, all_charts)

        tem_cond = 0
        for cc in NCi:
            tem_cond |= chart[:,cc]
        if np.sum(tem_cond) > 0:
            for ct,cv in enumerate(chart):
                if tem_cond[ct] and cv[0]:
                    pi.append(ct)

        CCall.extend(NCi)
        CCall_set.extend(NCi)
    CCall_set = list(set(CCall_set))
    print(f"size of Neg PI:{len(pi)}")
    print(f"CCall:{CCall_set}")

    for count, chart in enumerate(all_charts):
        chart = np.array(chart)
        tmp_CCi = DetUSMPosRuleSelect(count, all_charts)
        CCi = []
        for i in tmp_CCi:
            if i in CCall_set:
                CCi.append(i)
        tem_cond = 0
        for cc in CCi:
            tem_cond |= chart[:,cc]
        if np.sum(tem_cond) > 0:
            for ct,cv in enumerate(chart):
                if tem_cond[ct] and not cv[0]:
                    pi.append(ct)
    print(f"size of Neg + pos PI:{len(pi)}")
    return pi



        #for nc in NCi:
        #    ncs.extend(nc)
        #ncs = list(set(ncs))
        #CCall.extend(ncs)
        #print(f"class:{count}, NCs:{ncs}")

# ignore this "generate_prf1_curve" function
def generate_prf1_curve(all_charts):
    for count, chart in enumerate(all_charts):
        scores = []
        #chart = np.array(chart)
        for c, i in enumerate(confidences):
            pred = []
            true = []
            for d in chart:
                if d[0] == 1 and d[2+c] == True:
                    continue
                pred.append(d[0])
                true.append(d[1])
            print(f"chart:{len(chart)},pred:{len(pred)}, truth:{len(true)}")
            scores.append([i] + get_scores(true, pred))
        plt.figure()
        scores = np.array(scores)
        print(scores)
        plt.plot(scores[:,0],scores[:,1], color = 'r', label = "Precision")
        plt.plot(scores[:,0],scores[:,2], color = 'b', label = "Recall")
        plt.plot(scores[:,0],scores[:,3], color = 'k', label = "F1")
        plt.legend()
        plt.title(f"{count}_class")
        plt.savefig(f"{count}_class.png")
        plt.close()

all_charts = generate_chart(charts)
#print("Negative rules:")
#greedy_rules = GreedyNegRules(all_charts)
#print("Positive rules:")
#positive_rules = DetUSMPosRules(all_charts)
results = []
result0 = [0]
for count, chart in enumerate(all_charts):
    chart = np.array(chart)
    result0.extend(get_scores(chart[:,1],chart[:,0]))
    result0.extend([0,0, 0 ,0])
result0.extend(get_scores(true_data, pred_data))
results.append(result0)
epsilon = [0.001 * i for i in range(1, 100, 1)]

for ep in epsilon:
    #result = PosNegRuleLearn(all_charts, epsilon)
    result = ruleForNegativeCorrection(all_charts, ep)
    results.append([ep] + result)
    print(f"ep:{ep}\n{result}")
col = ['pre', 'recall', 'F1', 'NSC', 'PSC', 'NRC', 'PRC']
df = pd.DataFrame(results, columns = ['epsilon'] + col * n_classes + ['acc', 'macro-F1', 'micro-F1'])
df.to_csv(base_path0 + "rule_for_Negativecorrection.csv")

results = []
results.append(result0)
for ep in epsilon:
    #result = PosNegRuleLearn(all_charts, epsilon)
    result = ruleForNPCorrection(all_charts, ep)
    results.append([ep] + result)
    print(f"ep:{ep}\n{result}")
col = ['pre', 'recall', 'F1', 'NSC', 'PSC', 'NRC', 'PRC']
df = pd.DataFrame(results, columns = ['epsilon'] + col * n_classes + ['acc', 'macro-F1', 'micro-F1'])
df.to_csv(base_path0 + "rule_for_NPcorrection.csv")

results = []
results.append(result0)
for ep in epsilon:
    #result = PosNegRuleLearn(all_charts, epsilon)
    result = ruleForPNCorrection(all_charts, ep)
    results.append([ep] + result)
    print(f"ep:{ep}\n{result}")
df = pd.DataFrame(results, columns = ['epsilon'] + col * n_classes + ['acc', 'macro-F1', 'micro-F1'])
df.to_csv(base_path0 + "rule_for_PNcorrection.csv")

##generate_prf1_curve([charts])
#generate_prf1_curve(all_charts)

#np.save("chart_datas.npy", charts)
'''
{0, 1, 2, 3, 4, 5}
(9180, 5)
(9180, 5)
(9180, 5, 6)
(9180, 6, 5)
'''
'''
true_false = [[[] for j in range(len_labels)] for i in range(len_labels)]
[m,n] = true_data.shape
count = 0
max_sp0 = 0
for i in range(m):
    #print(true_data[i])
    for j in range(n):
        if true_data[i,j] == 0 :
            if max_sp0 < rawm_data[i,j,0]:
                max_sp0 = rawm_data[i,j,0]
        if true_data[i,j] != pred_data[i,j]:
            true_false[true_data[i,j]][pred_data[i,j]].append(orig_data[i,j])
            if pred_data[i,j] == 0 and rawm_data[i,j,0] > 0.25:
                print(f"{true_data[i,j]}, {rawm_data[i,j]}")
            count += 1
print(f"max speed for data 0 is : {max_sp0}")
'''        
print(f"Total true false data: {count}")
#for i in range(len_labels):
#    for j in range(len_labels):
#        print(f"##### {i} {j} {len(true_false[i][j])} ######")
#        print(true_false[i][j])
