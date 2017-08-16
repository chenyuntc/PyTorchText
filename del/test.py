#encoding:utf-8
from torch.utils import data
import torch as t
import numpy as np
from config import opt
import models
import json
import fire
import csv
from glob import glob
from torch.autograd import Variable
import tqdm

def load_data_stack(data_root,test,labels_file):
    '''
    data_root="/data/text/zhihu/result/*_test.pth"
    labels_file="/home/a/code/pytorch/zhihu/ddd/labels.json"
    test="/home/a/code/pytorch/zhihu/ddd/test.npz"
    '''
    result_files_path=glob(data_root+"*test.pth")
    index2qid=np.load(test)['index2qid'].item()
    with open(labels_file) as f:
        labels= json.load(f)['id2label']
    test_data_num=217360
    model_num=len(result_files_path)
    test_data=t.zeros(test_data_num,1999*model_num)
    for i in range(model_num):
        test_data[:,i*1999:i*1999+1999]=t.load(result_files_path[i]).float()
    return test_data,index2qid,labels
def load_data(type_='char'):
    with open(opt.labels_path) as f:
        labels_ = json.load(f)
    question_d = np.load(opt.test_data_path)
    if type_ == 'char':
        test_data_title,test_data_content =\
             question_d['title_char'],question_d['content_char']

    elif type_ == 'word':
        test_data_title,test_data_content =\
             question_d['title_word'],question_d['content_word']

    index2qid = question_d['index2qid'].item()
    return test_data_title,test_data_content,index2qid,labels_['id2label']
def write_csv(result,index2qid,labels):
    f=open(opt.result_path, "wa")
    csv_writer = csv.writer(f, dialect="excel")
    rows=[0 for _ in range(result.shape[0])]
    for i in range(result.shape[0]):
        row=[index2qid[i]]+[labels[str(int(i_))] for i_ in result[i]]
        rows[i]=row
    csv_writer.writerows(rows)
def dotest(model,title,content):
    title,content = Variable(t.from_numpy(title).long().cuda(),volatile=True),Variable(t.from_numpy(content).long().cuda(),volatile=True)
    score = model(title,content)
    probs=t.nn.functional.sigmoid(score)
    probs_ordered = probs.sort(dim=1,descending=True)
    tmp=probs_ordered[1][:,:5]
    return tmp.data.cpu().numpy()
    
def main(**kwargs):
    opt.parse(kwargs)
    model = getattr(models,opt.model)(opt).cuda().eval()
    if opt.model_path is not None:
        model.load(opt.model_path)
    test_data_title,test_data_content,index2qid,labels=load_data_stack(type_=opt.type_)
    Num=len(test_data_title)
    result=np.zeros((Num,5))
    for i in tqdm.tqdm(range(Num)):
        if i%opt.batch_size==0 and i>0:
            title=np.array(test_data_title[i-opt.batch_size:i])
            content=np.array(test_data_content[i-opt.batch_size:i])
            result[i-opt.batch_size:i,:]=dotest(model,title,content)  
        if i%opt.batch_size ==0:
            print i,"have done"
    title=np.array(test_data_title[opt.batch_size*(Num/opt.batch_size):])
    content=np.array(test_data_content[opt.batch_size*(Num/opt.batch_size):]) 
    result[opt.batch_size*(Num/opt.batch_size):,:]=dotest(model,title,content) 
    write_csv(result,index2qid,labels)
    
def main_stack(**kwargs):
    opt.parse(kwargs)
    model = getattr(models,opt.model)(opt).cuda().eval()
    if opt.model_path is not None:
        model.load(opt.model_path)
    data_root="/data/text/zhihu/result/"
    labels_file="/home/a/code/pytorch/zhihu/ddd/labels.json"
    test="/home/a/code/pytorch/zhihu/ddd/test.npz"
    test_data,index2qid,labels=load_data_stack(data_root,test,labels_file)
    Num=len(test_data)
    result=np.zeros((Num,5))
    for i in tqdm.tqdm(range(Num)):
        if i%opt.batch_size==0 and i>0:
            title=np.array(test_data_title[i-opt.batch_size:i])
            content=np.array(test_data_content[i-opt.batch_size:i])
            result[i-opt.batch_size:i,:]=dotest(model,title,content)  
        if i%opt.batch_size ==0:
            print i,"have done"
    title=np.array(test_data_title[opt.batch_size*(Num/opt.batch_size):])
    content=np.array(test_data_content[opt.batch_size*(Num/opt.batch_size):]) 
    result[opt.batch_size*(Num/opt.batch_size):,:]=dotest(model,title,content) 
    write_csv(result,index2qid,labels)
    

    
if __name__=='__main__':
    fire.Fire()
    
    