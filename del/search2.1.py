import json
import os
import sys
from utils import get_score
import torch as t 
import numpy as np
file = '/mnt/zhihu/data/val_best.pth'
r = t.load(file)
# b = t.load(file2)
# c = t.load(file3)
import time
test_data_path='/mnt/zhihu/data/val.npz'
index2qid = np.load(test_data_path)['index2qid'].item()
label_path='/mnt/zhihu/data/labels.json'
with open(label_path) as f: 
      labels_info = json.load(f)
qid2label = labels_info['d']
true_labels = [qid2label[index2qid[2999967-200000+ii]] for ii in range(len(r))]
len(true_labels)
previous_best_score=0.42
# large_index=[560, 753, 1208, 609, 741, 915, 879, 458, 1443, 1783]
large_index=[458, 1443, 1783]
def target(args):
        weight = args
        aaa = t.ones(1999)
        for ii,_ in enumerate(large_index):
              aaa[_] = args[ii]
      #   aaa[0],aaa[1],aaa[2],aaa[3],aaa[4] = args
        weight = aaa.view(1,-1).expand(200000,1999)
        r2 = weight*(r.float())
        result = r2.topk(5,1)[1]
        predict_label_and_marked_label_list = [[_1,_2] for _1,_2 in zip(result,true_labels)]
        score,_,_,_ = get_score(predict_label_and_marked_label_list)
      #   if score>previous_best_score:
        print score
            # previous_best_score = score
            # with open(str(score) ,'wb') as f:
            #     pickle.dump(args,f)
            
        return -score
from hyperopt import hp, fmin, rand, tpe, space_eval
list_space = [hp.uniform('b%s'%_,0.5,2) for _ in range(10)]
best = fmin(target,list_space,algo=tpe.suggest,max_evals=10)
best = fmin(target,list_space,algo=tpe.suggest,max_evals=10)
list_space = [hp.uniform('w1',0,2),hp.uniform('w2',0,2)]
best = fmin(target,list_space,algo=tpe.suggest,max_evals=50)
%hist -f search2.py


{'w0': 0.7261578854014094,
 'w1': 0.6729932326871956,
 'w2': 0.9624749042037957,
 'w3': 0.8998425892602284,
 'w4': 0.6488650207895496,
 'w5': 0.5219741148509414,
 'w6': 0.6845486024566358}

In [26]: val
Out[26]: 
['CNNText_tmp0.4024_char_val.pth',
 'DeepText0.4103_word_val.pth',
 'CNNText_tmp0.4109_word_val.pth',
 'LSTMText0.4119_word_val.pth',
 'RCNN_0.4037_char_val.pth',
 'LSTMText0.4031_char_val.pth',
 'RCNN_0.4115_word_val.pth']
