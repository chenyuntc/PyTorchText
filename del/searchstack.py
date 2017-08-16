import json
import os
import sys
from utils import get_score
import torch as t 
import numpy as np
from glob import glob
import pickle
max_evals=100
#data_root="/home/a/code/pytorch/zhihu/"
data_root="/data/text/zhihu/result/tmp/"
pre="LSTM"
files_path=glob(data_root+pre+"*val.pth")
files_path.sort()
model_num=len(files_path)
print files_path[:model_num]
probs=[t.load(r).float() for r in files_path[:model_num]]

test_data_path='/home/a/code/pytorch/zhihu/ddd/val.npz'
index2qid = np.load(test_data_path)['index2qid'].item()
label_path="/home/a/code/pytorch/zhihu/ddd/labels.json"
with open(label_path) as f: 
      labels_info = json.load(f)
qid2label = labels_info['d']
true_labels = [qid2label[index2qid[2999967-200000+ii]] for ii in range(len(probs[0]))]
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

from hyperopt import hp, fmin, rand, tpe, space_eval
list_space = [hp.normal('a'+str(rr),1,0.2) for rr in range(model_num)]
from hyperopt import Trials
trials_to_keep=Trials()
best = fmin(target,list_space,algo=tpe.suggest,max_evals=max_evals, trials = trials_to_keep)
output = open('trials_to_keep__'+pre+'.pkl', 'wb')
pickle.dump(trials_to_keep, output)

    


