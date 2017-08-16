#encoding:utf-8
from torch.utils import data
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
    with open(opt.labels_path) as f:
        labels_ = json.load(f)
    print "data_path: ",opt.test_data_path
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
    return probs.data.cpu().numpy()
    
def main(**kwargs):
    opt.parse(kwargs)
    model = getattr(models,opt.model)(opt).cuda().eval()
    if opt.model_path is not None:
        model.load(opt.model_path)
    opt.parse(kwargs)
    
    model = model.eval()
    
    test_data_title,test_data_content,index2qid,labels=load_data(type_=opt.type_)
    Num=len(test_data_title)
    print "Num: ",Num
    result=np.zeros((Num,1999))
    for i in tqdm.tqdm(range(Num)):
        if i%opt.batch_size==0 and i>0:
            # import ipdb;ipdb.set_trace()
            title=np.array(test_data_title[i-opt.batch_size:i])
            content=np.array(test_data_content[i-opt.batch_size:i])
            result[i-opt.batch_size:i,:]=dotest(model,title,content)  
    if Num%opt.batch_size!=0:
        title=np.array(test_data_title[opt.batch_size*(Num/opt.batch_size):])
        content=np.array(test_data_content[opt.batch_size*(Num/opt.batch_size):]) 
        result[opt.batch_size*(Num/opt.batch_size):,:]=dotest(model,title,content) 
    t.save(t.from_numpy(result).float(),opt.result_path)
    
if __name__=='__main__':
    fire.Fire()
    
    