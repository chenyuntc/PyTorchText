import json
import os
import sys
from utils import get_score
import torch as t 
import numpy as np
from glob import glob
import pickle
from glob import glob
data_root="/data_ssd/zhihu/result/tmp/"
files_path=glob(data_root+"*val.pth")
files_path.sort()
print len(files_path)
files_weight1=[]
initial_weight=[]
for file in files_path:
    if 'weight5' in file:
        files_weight1.append(file)
        if 'MultiModel' in file:
            initial_weight.append(5)
        else:
            initial_weight.append(1)
print len(files_weight1)
for f,w in zip(files_weight1,initial_weight):
    print f,w
model_num=10 if len(files_weight1)>10 else len(files_weight1) 
probs=[t.load(r).float() for r in files_weight1[:model_num]]
test_data_path='/home/a/code/pytorch/zhihu/ddd/val.npz'
index2qid = np.load(test_data_path)['index2qid'].item()
label_path="/home/a/code/pytorch/zhihu/ddd/labels.json"
with open(label_path) as f: 
      labels_info = json.load(f)
qid2label = labels_info['d']
true_labels = [qid2label[index2qid[2999967-200000+ii]] for ii in range(200000)]
del labels_info
del qid2label
del index2qid
def target(args):
    r=0
    for r_,k_ in enumerate(args):
        if r_<model_num:
            r +=k_*probs[r_]
        else:
            tmp=t.load(files_path[r_]).cuda().float()
            r=r+k_*tmp.cpu()
    result = r.topk(5,1)[1]
    predict_label_and_marked_label_list = [[_1,_2] for _1,_2 in zip(result,true_labels)]
    score,_,_,rrs = get_score(predict_label_and_marked_label_list)
    print (args,score,rrs)#list_space = [hp.uniform('a',0,1),hp.uniform('b',0,1)]
    return -rrs[0]
max_evals=150
from hyperopt import hp, fmin, rand, tpe, space_eval
list_space = [hp.normal('a'+str(rr),initial_weight[rr],0.2) for rr in range(len(files_weight1))]
from hyperopt import Trials
trials_to_keep=Trials()
best = fmin(target,list_space,algo=tpe.suggest,max_evals=max_evals, trials = trials_to_keep)
output = open('trials_to_keep__weight5_only'+'.pkl', 'wb')
pickle.dump(trials_to_keep, output)
output.close()


print best
results=0
result_path="/data_ssd/zhihu/result/search_result/"
for ii in range(len(files_weight1)):
    if ii<model_num:
        results +=probs[ii]*best['a'+str(ii)]
    else:
        tmp=t.load(files_path[ii]).cuda().float()
        results +=tmp*best['a'+str(ii)]
t.save(results.float(),result_path+"search_weight5_result_first.pth")


