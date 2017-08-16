#coding:utf8
import sys
sys.path.append('../')
from utils import get_score
import json
import pickle

file1='/mnt/zhihu/data/RCNN_deep_word_val_4115'
file2='/mnt/zhihu/data/rccndeep_char_val_4037.pth'
file3='/mnt/zhihu/data/multicnntextbndeep40705_val_word.pth'

label_path =   '/mnt/zhihu/data/labels.json'
# test_data_path='/mnt/zhihu/data/test.npz'
def ensamble(file1,file2,file3,label_path=label_path,test_data_path=test_data_path,result_csv=None):
    import torch as t 
    import numpy as np
    if result_csv is None:
        import time
        result_csv = time.strftime('%y%m%d_%H%M%S.csv')
    a = t.load(file1)
    b = t.load(file2)
    c = t.load(file3)

    index2qid = np.load(test_data_path)['index2qid'].item()
    with open(label_path) as f: 
          labels_info = json.load(f)
    qid2label = labels_info['d']
    # with open(label_path) as f:   label2qid = json.load(f)['id2label']
    true_labels = [qid2label[index2qid[2999967-200000+ii]] for ii in range(len(a))]
    # for ii,item in enumerate(result):
    #     rows[ii] = [index2qid[ii]] + [label2qid[str(_)] for _ in item ]

    previous_best_score = 0.42
    def target(args):
        w1,w2 = args
        r = a + b*w1 +c*w2
        result = r.topk(5,1)[1]
        predict_label_and_marked_label_list = [[_1,_2] for _1,_2 in zip(result,true_labels)]
        score,_,_,_ = get_score(predict_label_and_marked_label_list)
        print (args,score,_)
        if score>previous_best_score:
            previous_best_score = score
            with open(str(score) ,'wb') as f:
                pickle.dump(args,f)
             
        return -score
    list_space = [hp.uniform('w1',0,2),hp.uniform('w2',0,2)]
    best = fmin(new_target,list_space,algo=tpe.suggest,max_evals=50)
    print best
    # import csv
    # with open(result_csv,'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(rows)




if __name__ == '__main__':
    import fire
    fire.Fire()
