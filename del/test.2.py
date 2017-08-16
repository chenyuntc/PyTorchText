#encoding:utf-8
'''
直接根据模型生成可以提交的csv
专门为multimodel 而写
dataset同时送进去char和word
'''
from torch.utils import data
import torch as t
import numpy as np
from config import opt
import models
import json
import fire
import csv
from torch.autograd import Variable
import tqdm

def load_data(type_='char'):
    with open(opt.labels_path) as f:
        labels_ = json.load(f)
    question_d = np.load(opt.test_data_path)
    # if type_ == 'char':
    #     test_data_title,test_data_content =\
    #          question_d['title_char'],question_d['content_char']

    # elif type_ == 'word':
    #     test_data_title,test_data_content =\
    #          question_d['title_word'],question_d['content_word']

    index2qid = question_d['index2qid'].item()

    return (question_d['title_word'],question_d['content_word']),(question_d['title_char'],question_d['content_char']),index2qid,labels_['id2label']
def write_csv(result,index2qid,labels):
    f=open(opt.result_path, "wa")
    csv_writer = csv.writer(f, dialect="excel")
    rows=[0 for _ in range(result.shape[0])]
    for i in range(result.shape[0]):
        row=[index2qid[i]]+[labels[str(int(i_))] for i_ in result[i]]
        rows[i]=row
    csv_writer.writerows(rows)
def dotest(model,title,content):
    title,content = (Variable(t.from_numpy(title[0]).long().cuda(),volatile=True), Variable(t.from_numpy(title[1]).long().cuda(),volatile=True)),(Variable(t.from_numpy(content[0]).long().cuda(),volatile=True),Variable(t.from_numpy(content[1]).long().cuda(),volatile=True))
    score = model(content,title)
    probs=t.nn.functional.sigmoid(score)
    probs_ordered = probs.sort(dim=1,descending=True)
    tmp=probs_ordered[1][:,:5]
    return tmp.data.cpu().numpy()
    
def main(**kwargs):
    opt.parse(kwargs)
    # opt.model_names=['MultiCNNTextBNDeep','RCNN','LSTMText','CNNText_inception','RCNN','CNNText_inception','LSTMText']
    # opt.model_paths = ['checkpoints/MultiCNNTextBNDeep_word_0.41124002492','checkpoints/RCNN_word_0.411511574999','checkpoints/LSTMText_word_0.411994005382','checkpoints/CNNText_tmp_char_0.402429167301','checkpoints/RCNN_char_0.403710422571','checkpoints/CNNText_tmp_word_0.41096749885','checkpoints/LSTMText_char_0.403192339135',]
    model = getattr(models,opt.model)(opt).cuda().eval()
    if opt.model_path is not None:
        model.load(opt.model_path)
    model=model.eval()
    test_data_title,test_data_content,index2qid,labels=load_data(type_=opt.type_)
    Num=len(test_data_title[0])
    result=np.zeros((Num,5))
    for i in tqdm.tqdm(range(Num)):
        if i%opt.batch_size==0 and i>0:
            title=np.array(test_data_title[0][i-opt.batch_size:i]),np.array(test_data_title[1][i-opt.batch_size:i])
            content=np.array(test_data_content[0][i-opt.batch_size:i]),np.array(test_data_content[1][i-opt.batch_size:i])
            result[i-opt.batch_size:i,:]=dotest(model,title,content)  
        # if i%opt.batch_size ==0:
        #     print i,"have done"


    result[opt.batch_size*(Num/opt.batch_size):,:]=dotest(model,title,content) 
    write_csv(result,index2qid,labels)
    

    

    
if __name__=='__main__':
    fire.Fire()
    
    