import json
import os
import sys
from utils import get_score
import torch as t 
import numpy as np
file1='/mnt/zhihu/data/rcnndeep4115_word_val.pth'
file2='/mnt/zhihu/data/rccndeep4037_char_val.pth'
file3='/mnt/zhihu/data/multicnntextbndeep40705_word_val.pth'
a = t.sigmoid(t.load(file1).float())
b = t.sigmoid(t.load(file2).float())
c = t.sigmoid(t.load(file3).float())
import time
test_data_path='/mnt/zhihu/data/val.npz'
index2qid = np.load(test_data_path)['index2qid'].item()
label_path='/mnt/zhihu/data/labels.json'
with open(label_path) as f: 
      labels_info = json.load(f)
qid2label = labels_info['d']
true_labels = [qid2label[index2qid[2999967-200000+ii]] for ii in range(len(a))]
len(true_labels)
def target(args):
        w1,w2,w3 = args
        r = a + b*w1 +c*w2 + d*w3
        result = r.topk(5,1)[1]
        predict_label_and_marked_label_list = [[_1,_2] for _1,_2 in zip(result,true_labels)]
        score,_,_,_ = get_score(predict_label_and_marked_label_list)
        print (args,score,_)#list_space = [hp.uniform('a',0,1),hp.uniform('b',0,1)]
        return -score
from hyperopt import hp, fmin, rand, tpe, space_eval
list_space = [hp.uniform('a',0,1),hp.uniform('b',0,1)]
best = fmin(target,list_space,algo=tpe.suggest,max_evals=10)
best = fmin(target,list_space,algo=tpe.suggest,max_evals=10)
list_space = [hp.uniform('w1',0,2),hp.uniform('w2',0,2)]
best = fmin(new_target,list_space,algo=tpe.suggest,max_evals=50)
%hist -f search2.py
# ((0.9737992669297721, 0.17478779579719494), 0.4223127848108588, [116814, 64995, 42622, 29421, 21724])
# 0.42238923442022624,
#((0.8951200869708624, 0.13488758907249945), 0.42242919538129015, [116791, 64994, 42467, 29659, 21763])
# ([0.9903428611407182, 0.8213389405505421, 1.032499220786336], 0.4241382314016921, [117305, 65592, 42492, 29488, 21876])
