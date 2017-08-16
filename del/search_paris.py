import json
import os
import sys
from utils import get_score
import torch as t 
import numpy as np
from glob import glob
import pickle
from glob import glob
pre="search_nomulti_weight1"
data_root="/data_ssd/zhihu/result/search_result/"
files_path=glob(data_root+pre+"*.pth")
files_path.sort()
print files_path
probs=[t.load(r).float() for r in files_path]
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
    for r_,k_ in zip(args,probs):
        r=r+r_*k_
    result = r.topk(5,1)[1]
    predict_label_and_marked_label_list = [[_1,_2] for _1,_2 in zip(result,true_labels)]
    score,_,_,_ = get_score(predict_label_and_marked_label_list)
    print (args,score,_)#list_space = [hp.uniform('a',0,1),hp.uniform('b',0,1)]
    return -score
max_evals=50
from hyperopt import hp, fmin, rand, tpe, space_eval
list_space = [hp.normal('a'+str(rr),1,0.5) for rr in range(len(files_path))]
from hyperopt import Trials
trials_to_keep=Trials()
best = fmin(target,list_space,algo=tpe.suggest,max_evals=max_evals, trials = trials_to_keep)
output = open('trials_to_keep__'+pre+'pairs_.pkl', 'wb')
pickle.dump(trials_to_keep, output)
output.close()


print best
results=0
result_path="/data_ssd/zhihu/result/search_result/"
for ii in range(len(files_path)):
    results +=probs[ii]*best['a'+str(ii)]
t.save(results.float(),result_path+pre+"_pairs.pth")




    