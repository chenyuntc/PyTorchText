#encoding:utf-8
# from torch.utils import data
import torch as t
import numpy as np
from config import opt
import models
import json
import fire
import csv
import tqdm
from torch.autograd import Variable
def load_data(type_='char'):
    id2label = t.load(opt.id2label)
    question_d = np.load(opt.test_data_path)
    index2qid = question_d['index2qid'].item()
    return (question_d['title_char'],question_d['content_char']),( question_d['title_word'],question_d['content_word']),index2qid,id2label
def write_csv(result,index2qid,labels):
    f=open(opt.result_path, "wa")
    csv_writer = csv.writer(f, dialect="excel")
    rows=[0 for _ in range(result.shape[0])]
    for i in range(result.shape[0]):
        row=[index2qid[i]]+[labels[str(int(i_))] for i_ in result[i]]
        rows[i]=row
    csv_writer.writerows(rows)
def dotest(model,title,content):
    title,content = (Variable(t.from_numpy(title[0]).long().cuda(),volatile=True),Variable(t.from_numpy(title[1]).long().cuda(),volatile=True)),(Variable(t.from_numpy(content[0]).long().cuda(),volatile=True),Variable(t.from_numpy(content[1]).long().cuda(),volatile=True))
    score = model(title,content)
    probs=t.nn.functional.sigmoid(score)
    return probs.data.cpu().numpy()
    
def test_one(file,data_):
    test_data_title,test_data_content,index2qid,labels = data_
    opt.model_path = file
    model= models.MultiModelAll4zhihu(opt).cuda().eval()
    
    Num=len(test_data_title[0])
    result=np.zeros((Num,1999))
    for i in tqdm.tqdm(range(Num)):
        if i%opt.batch_size==0 and i>0:
            title=np.array(test_data_title[0][i-opt.batch_size:i]),np.array(test_data_title[1][i-opt.batch_size:i])
            content=np.array(test_data_content[0][i-opt.batch_size:i]),np.array(test_data_content[1][i-opt.batch_size:i])
            result[i-opt.batch_size:i,:]=dotest(model,title,content)  
    if Num%opt.batch_size!=0:
        title=np.array(test_data_title[0][opt.batch_size*(Num/opt.batch_size):]),np.array(test_data_title[1][opt.batch_size*(Num/opt.batch_size):])
        content=np.array(test_data_content[0][opt.batch_size*(Num/opt.batch_size):]) ,np.array(test_data_content[1][opt.batch_size*(Num/opt.batch_size):]) 
        result[opt.batch_size*(Num/opt.batch_size):,:]=dotest(model,title,content) 
    # t.save(t.from_numpy(result).float(),opt.result_path)
    return t.from_numpy(result).float()

if __name__=='__main__':
    
    # 如果路径不一致记得修改这里
    opt.result_path = 'submission.csv'
    opt.test_data_path='test.npz'
    opt.id2label='id2label.json'
    
    data_=load_data(type_='all')
    test_data_title,test_data_content,index2qid,labels = data_ 
    label2qid = labels
    # 如果路径不一致记得修改这里
    model_paths = [  
            'MultiModelAll_word_0.416523282174',
            'MultiModelAll_word_0.417185977233',
            'MultiModelAll_word_0.419866393964',
            'MultiModelAll_word_0.421331405593',
            'MultiModelAll_word_0.421692025324',
            'MultiModelAll_word_0.423535867989',
            'MultiModelAll_word_0.419245894992'
            ] 
    
    result= 0


    for model_path in model_paths:
        result += test_one(model_path,data_)
    result_labels=(result).topk(5,1)[1]

    ## 写csv 提交结果
    rows = range(result.size(0))
    for ii,item in enumerate(result_labels):
        rows[ii] = [index2qid[ii]] + [label2qid[str(_)] for _ in item ]
    import csv
    with open(opt.result_path,'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        
        
