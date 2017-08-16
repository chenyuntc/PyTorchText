#coding:utf-8
import json
import os
import sys
from utils import get_score
import torch as t 
import numpy as np
from glob import glob
import pickle
from glob import glob
noaug=True
no_mul_w1=True
multimodel=True
weight5=True
allmoel=True
pre="first"
def write_csv(result,index2qid,labels,filename):
    path="/data_ssd/zhihu/result/search_result/"
    f=open(path+filename+".csv", "wa")
    csv_writer = csv.writer(f, dialect="excel")
    rows=[0 for _ in range(result.size(0))]
    for i in range(result.size(0)):
        tmp=result[i].sort(dim=0,descending=True)
        tmp=tmp[1][:5]
        row=[index2qid[i]]+[labels[str(int(i_))] for i_ in tmp]
        rows[i]=row
    csv_writer.writerows(rows) 
def load_data(test,labels_file):
    index2qid=np.load(test)['index2qid'].item()
    with open(labels_file) as f:
        labels= json.load(f)['id2label']
    return index2qid,labels
#labels_file="/home/a/code/pytorch/zhihu/ddd/labels.json"
#test="/home/a/code/pytorch/zhihu/ddd/test.npz"
#index2qid,labels=load_data(test,labels_file)
################################################
#no aug  result目录下非增强测试
if noaug:
    data_root="/data_ssd/zhihu/result/"
    files_path=glob(data_root+"*val.pth")
    files_path.sort()
    files=[]
    for file in files_path:
         if 'MultiModel' not in file:
            files.append(file)        
    print len(files_path)
    pkl_file=open("trials_to_keep__weight1_noaug.pkl",'rb')
    trials=pickle.load(pkl_file)
    model_num=len(files)
    params=[trials.best_trial['misc']['vals']['a'+str(rr)]for rr in range(8)]
    result_1=0
    for i,pa in enumerate (params):
            print "load ",files[i],pa
            result_1=result_1+pa[0]*t.load(files[i].replace("val","test")).float()
    t.save(result_1.float(),"/data_ssd/zhihu/result/search_result/test_noaug_first.pth")
    #write_csv(result,index2qid,labels,"test_noaug")
################################################
# aug  no multimodel weight1   目录下非增强测试
if no_mul_w1:
    data_root="/data_ssd/zhihu/result/tmp/"
    files_path=glob(data_root+"*val.pth")
    files_path.sort()
    files=[]
    for file in files_path:
        if 'MultiModel' not in file and 'weight5' not in file:
            files.append(file)        
    print files
    print len(files)
    pkl_file=open("trials_to_keep__aug_nomulti_weight1.pkl",'rb')
    trials=pickle.load(pkl_file)
    model_num=len(files)
    params=[trials.best_trial['misc']['vals']['a'+str(rr)]for rr in range(len(files))]
    result_1=0
    for i,pa in enumerate (params):
            print "load ",files[i],pa
            result_1=result_1+pa[0]*t.load(files[i].replace("val","test")).float()
    #write_csv(result,index2qid,labels,"test_no_mul_w1")
    t.save(result_1.float(),"/data_ssd/zhihu/result/search_result/test_test_nomultiw1_first.pth")
################################################
# multimodel    目录下非增强测试
if multimodel:
    data_root="/data_ssd/zhihu/result/tmp/"
    files_path=glob(data_root+"*val.pth")
    files_path.sort()
    files=[]
    for file in files_path:
        if 'MultiModel' in file: 
            files.append(file)        
    print len(files)
    pkl_file=open("trials_to_keep_multimodel.pkl",'rb')
    trials=pickle.load(pkl_file)
    model_num=len(files)
    params=[trials.best_trial['misc']['vals']['a'+str(rr)]for rr in range(len(files))]
    result_1=0
    for i,pa in enumerate (params):
            print "load ",files[i],pa
            result_1=result_1+pa[0]*t.load(files[i].replace("val","test")).float()
    t.save(result_1.float(),"/data_ssd/zhihu/result/search_result/test_multimodel_first.pth")
    #write_csv(result,index2qid,labels,"test_multimodel")

################################################
# weight5    目录下非增强测试
if weight5:
    data_root="/data_ssd/zhihu/result/tmp/"
    files_path=glob(data_root+"*val.pth")
    files_path.sort()
    files=[]
    for file in files_path:
        if 'weight5' in file:
            files.append(file)
    print len(files)
    pkl_file=open("trials_to_keep__weight5_only.pkl",'rb')
    trials=pickle.load(pkl_file)
    model_num=len(files)
    params=[trials.best_trial['misc']['vals']['a'+str(rr)]for rr in range(len(files))]
    result_1=0
    for i,pa in enumerate (params):
            print "load ",files[i],pa
            result_1=result_1+pa[0]*t.load(files[i].replace("val","test")).float()
    t.save(result_1.float(),"/data_ssd/zhihu/result/search_result/test_weight5_first.pth")
    #write_csv(result,index2qid,labels,"test_weight5") 
    
    
################################################
# allmodel    四大类融合
if allmoel:
    data_root="/data_ssd/zhihu/result/search_result/"
    files_path=glob(data_root+"test*first.pth")
    files_path.sort()
    files= files_path      
    print len(files_path)
    pkl_file=open("trials_to_keep_all_first.pkl",'rb')
    trials=pickle.load(pkl_file)
    model_num=len(files)
    params=[trials.best_trial['misc']['vals']['a'+str(rr)]for rr in range(len(files))]
    result_1=0
    for i,pa in enumerate (params):
            print "load ",files[i],pa
            result_1=result_1+pa[0]*t.load(files[i]).float()
    t.save(result_1.float(),"/data_ssd/zhihu/result/search_result/test_all_first.pth")
    #write_csv(result,index2qid,labels,"test_weight5") 



