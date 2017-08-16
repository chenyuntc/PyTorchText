#encoding:utf-8

'''
测试时对数据进行增强，分别元数据 shuffle  drop三种
情况下结果，通过hyper search进行融合
'''
from utils import get_score
from torch.utils import data
import torch as t
import numpy as np
from config import opt
import models
import json
import os
import sys
import fire
from glob import glob
import csv
import tqdm
import pickle
from torch.autograd import Variable
def load_data(type_='char'):
    with open(opt.labels_path) as f:
        labels_ = json.load(f)
    question_d = np.load(opt.test_data_path)
    print "test_path: ",opt.test_data_path
    index2qid = question_d['index2qid'].item()
    index=np.arange(len(question_d['title_char']))
    np.random.shuffle(index)
    char_data=(question_d['title_char'],question_d['content_char'])
    word_data=(question_d['title_word'],question_d['content_word'])
    return  char_data,word_data,index2qid,labels_,index
def write_csv(result,index2qid,labels):
    f=open(opt.result_path, "wa")
    csv_writer = csv.writer(f, dialect="excel")
    rows=[0 for _ in range(result.size(0))]
    for i in range(result.size(0)):
        tmp=result[i].sort(dim=0,descending=True)
        tmp=tmp[1][:5]
        row=[index2qid[i]]+[labels[str(int(i_))] for i_ in tmp]
        rows[i]=row
    csv_writer.writerows(rows)  


def dotest(model,title,content):
    title,content = (Variable(t.from_numpy(title[0]).long().cuda(),volatile=True),Variable(t.from_numpy(title[1]).long().cuda(),volatile=True)),(Variable(t.from_numpy(content[0]).long().cuda(),volatile=True),Variable(t.from_numpy(content[1]).long().cuda(),volatile=True))
    score = model(title,content)
    probs=t.nn.functional.sigmoid(score)
    return probs.data.cpu().numpy()
def dropout(d,p=0.5):
        len_ = d.shape[1]
        batch_=d.shape[0]
        for ii in range(batch_):
            index = np.random.choice(len_,int(len_*p))
            d[ii,index]=0
        return d 
def test_val():
    #####
    #####测试增强前的验证集结果
    #####
    data_path='/data_ssd/zhihu/result/tmp/'
    result_path=['inception0.41254_aug_word_val.pth','LSTMText0.41368_aug_word_val.pth','DeepText0.38738_aug_char_val.pth',
                 ' RCNN0.39854_aug_char_val.pth','RCNN0.41344_aug_word_val.pth']#,'LSTMText0.4120_aug_word_val.pth']
    result=0
    for i in range(len(result_path)):
        path=data_path+result_path[i]
        print "loading",path
        result +=t.load(path.replace(' ','')).float()
    test_data_path='/home/a/code/pytorch/zhihu/ddd/val.npz'
    index2qid = np.load(test_data_path)['index2qid'].item()
    label_path="/home/a/code/pytorch/zhihu/ddd/labels.json"
    with open(label_path) as f: 
          labels_info = json.load(f)
    qid2label = labels_info['d']
    true_labels = [qid2label[index2qid[2999967-200000+ii]] for ii in range(200000)]
    result_ = result.topk(5,1)[1]
    predict_label_and_marked_label_list = [[_1,_2] for _1,_2 in zip(result_,true_labels)]
    score,_,_,_ = get_score(predict_label_and_marked_label_list)
    print (score,_)
def test_val_aug():
    #####
    #####测试增强后的验证集结果
    #####
    data_path="/data_ssd/zhihu/result/test_aug/"
    result_path=glob(data_path+"*.npz")
    result_path.sort()
    result=0
    for i in range(len(result_path)):
        print "loading",result_path[i]
        result +=np.load(result_path[i])['result_prob']
    test_data_path='/home/a/code/pytorch/zhihu/ddd/val.npz'
    index2qid = np.load(test_data_path)['index2qid'].item()
    label_path="/home/a/code/pytorch/zhihu/ddd/labels.json"
    with open(label_path) as f: 
          labels_info = json.load(f)
    qid2label = labels_info['d']
    true_labels = [qid2label[index2qid[2999967-200000+ii]] for ii in range(200000)]
    result_ = t.from_numpy(result).topk(5,1)[1]
    predict_label_and_marked_label_list = [[_1,_2] for _1,_2 in zip(result_,true_labels)]
    score,_,_,_ = get_score(predict_label_and_marked_label_list)
    print (score,_)

    
def main(**kwargs):
    opt.parse(kwargs)
    ####################### MultiModelAll_word_0.417185977233#################
    opt.model_names=['MultiCNNTextBNDeep','LSTMText','CNNText_inception','RCNN']
    opt.model_paths = ['checkpoints/MultiCNNTextBNDeep_0.37125473788','checkpoints/LSTMText_word_0.381833388089','checkpoints/CNNText_tmp_0.376364647145','checkpoints/RCNN_char_0.3456599248']
    #########################################################################
    
    
    model = getattr(models,opt.model)(opt).cuda().eval()
    if opt.model_path is not None:
        model.load(opt.model_path)
    opt.parse(kwargs)
    
    model = model.eval()
    
    test_data_title,test_data_content,index2qid,labels,index=load_data(type_=opt.type_)
    Num=len(test_data_title[0])
    print "Num: ",Num
    result1=np.zeros((Num,1999))
    result2=np.zeros((Num,1999))
    result3=np.zeros((Num,1999))
    for i in tqdm.tqdm(range(Num)):
        if i%opt.batch_size==0 and i>0:
            # import ipdb;ipdb.set_trace()
            title1=np.array(test_data_title[0][i-opt.batch_size:i]),np.array(test_data_title[1][i-opt.batch_size:i])
            content1=np.array(test_data_content[0][i-opt.batch_size:i]),np.array(test_data_content[1][i-opt.batch_size:i])
            result1[i-opt.batch_size:i,:]=dotest(model,title1,content1) 
            
            title3=dropout(title1[0],0.3),dropout(title1[1],0.3)
            content3=dropout(content1[0],0.7),dropout(content1[1],0.7)
            result3[i-opt.batch_size:i,:]=dotest(model,title3,content3)
            
            title2=np.array(test_data_title[0][index[i-opt.batch_size:i]]),np.array(test_data_title[1][index[i-opt.batch_size:i]])
            content2=np.array(test_data_content[0][index[i-opt.batch_size:i]]),np.array(test_data_content[1][index[i-opt.batch_size:i]])
            result2[index[i-opt.batch_size:i]]=dotest(model,title2,content2)
            
    if Num%opt.batch_size!=0:
        title1=np.array(test_data_title[0][opt.batch_size*(Num/opt.batch_size):]),np.array(test_data_title[1][opt.batch_size*(Num/opt.batch_size):])
        content1=np.array(test_data_content[0][opt.batch_size*(Num/opt.batch_size):]),np.array(test_data_content[1][opt.batch_size*(Num/opt.batch_size):])
        result1[opt.batch_size*(Num/opt.batch_size):,:]=dotest(model,title1,content1) 
        
        title3=dropout(title1[0],0.3),dropout(title1[1],0.3)
        content3=dropout(content1[0],0.7),dropout(content1[1],0.7)
        result3[opt.batch_size*(Num/opt.batch_size):]=dotest(model,title3,content3)

        title2=np.array(test_data_title[0][index[opt.batch_size*(Num/opt.batch_size):]]),np.array(test_data_title[1][index[opt.batch_size*(Num/opt.batch_size):]])
        content2=np.array(test_data_content[0][index[opt.batch_size*(Num/opt.batch_size):]]),np.array(test_data_content[1][index[opt.batch_size*(Num/opt.batch_size):]])
        result2[index[opt.batch_size*(Num/opt.batch_size):]]=dotest(model,title2,content2)
        
        
        
    #######
    #hyper search
    ######
    if opt.val:
        probs=[result1,result2,result3]
        def target(args):
            r=0
            for r_,k_ in zip(args,probs):
                r=r+r_*k_
            result = t.from_numpy(r).topk(5,1)[1]
            predict_label_and_marked_label_list = [[_1,_2] for _1,_2 in zip(result,true_labels)]
            score,_,_,_ = get_score(predict_label_and_marked_label_list)
            print (args,score,_)#list_space = [hp.uniform('a',0,1),hp.uniform('b',0,1)]
            return -score


        true_labels = [labels['d'][index2qid[2999967-200000+ii]] for ii in range(200000)]
        from hyperopt import hp, fmin, rand, tpe, space_eval
        list_space = [hp.normal('a0',1,0.8),hp.normal('a1',0.5,0.3),hp.normal('a2',0.2,0.2)]
        from hyperopt import Trials
        trials_to_keep=Trials()
        best = fmin(target,list_space,algo=tpe.suggest,max_evals=100, trials = trials_to_keep)
        best_params=[best['a0'],best['a1'],best['a2']]
        sums=best['a0']+best['a1']+best['a2']
        best_score=trials_to_keep.best_trial['result']['loss']*(-1)
        result=(result1*best['a0']+result2*best['a1']+result3*best['a2'])/sums
        output = open('trials_to_keep_'+opt.model+"_"+str(best_score)+'.pkl', 'wb')
        pickle.dump(trials_to_keep, output)
        npz_file='/data_ssd/zhihu/result/test_aug/'+opt.model+"_"+str(best_score)+'.npz'
        np.savez_compressed(npz_file, weights=np.array(best_params), result_prob=result)
    else:
        a1=1.08507012#1.1419278470112755
        a2=-1.04290239#-1.0272661867201016
        a3=1.16344534#1.1450416449670469
        result=(a1*result1+a2*result2+a3*result3)/(a1+a2+a3)
        labels__=labels['id2label']
        #result_top=t.from_numpy(result).topk(5,1)[1]
        write_csv(t.from_numpy(result),index2qid,labels__)
    #t.save(t.from_numpy(result1).float(),"test_aug_1.pth")
    #t.save(t.from_numpy(result2).float(),"test_aug_2.pth")
    #t.save(t.from_numpy(result3).float(),"test_aug_3.pth")
    #if opt.val:
    #    true_labels = [labels[index2qid[2999967-200000+ii]] for ii in range(len(test_data_title))]
    #    result_ = t.from_numpy(result).topk(5,1)[1]
    #    predict_label_and_marked_label_list = [[_1,_2] for _1,_2 in zip(result_,true_labels)]
    #    score,_,_,_ = get_score(predict_label_and_marked_label_list)
    #    print score
    del result1
    del result2
    del result3
    del result
    
    
if __name__=='__main__':
    fire.Fire()
    
    