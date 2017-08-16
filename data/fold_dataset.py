#encoding:utf-8
from torch.utils import data
import torch as t
import numpy as np
import random
from glob import glob
import json
    
    
class FoldData(data.Dataset):
    '''没什么用'''

    def __init__(self,train_root,labels_file,type_='char',fold=0):
        '''
        Dataset('/mnt/7/zhihu/ieee_zhihu_cup/train.npz','/mnt/7/zhihu/ieee_zhihu_cup/a.json')
        '''
        import json
        with open(labels_file) as f:
            labels_ = json.load(f)
        self.fold =fold

        # embedding_d = np.load(embedding_root)['vector']
        question_d = np.load(train_root)
        self.type_=type_
        if type_ == 'char':
            all_data_title,all_data_content =\
                 question_d['title_char'],question_d['content_char']

        elif type_ == 'word':
            all_data_title,all_data_content =\
                 question_d['title_word'],question_d['content_word']

        self.train_data = all_data_title[:-200000],all_data_content[:-200000]
        self.val_data = all_data_title[-200000:],all_data_content[-200000:]

        self.all_num = len(all_data_content)
        # del all_data_title,all_data_content
        
        self.data_title,self.data_content = self.train_data
        self.len_ = len(self.data_title)
        
        self.training=True

        self.index2qid = question_d['index2qid'].item()
        self.l_end=0
        self.labels = labels_['d']

    def shuffle(self,d):
        return np.random.permutation(d.tolist())

    def dropout(self,d,p=0.5):
        len_ = len(d)
        index = np.random.choice(len_,int(len_*p))
        d[index]=0
        return d     

    def train(self, train=True):
        if train:
            self.training=True
            self.data_title,self.data_content = self.train_data
            self.l_end = 0
        else:
            self.training=False
            self.data_title,self.data_content = self.val_data
            self.l_end = self.all_num-200000
        self.len_ = len(self.data_content)
        return self

    def __getitem__(self,index):
        '''
        for (title,content),label in dataloader:
            train
`       
        当使用char时
        title: (50,)
        content: (250,)
        labels：(1999,)
        '''
        augument_type=0
        if self.training: 
            index  = int(index/2)*2+self.fold
            index = index%self.len_
            augument=(index%2)        

        title,content =  self.data_title[index],self.data_content[index]
        qid = self.index2qid[index+self.l_end]
        labels = self.labels[qid]
        if self.training :
            if augument==0:
                title = self.dropout(title,p=0.3)
                content = self.dropout(content,p=0.7)
            else:
                title = self.shuffle(title)
                content = self.shuffle(content)
        data = (t.from_numpy(title).long(),t.from_numpy(content).long())
        label_tensor = t.zeros(1999).scatter_(0,t.LongTensor(labels),1).long()
        return data,label_tensor

    def __len__(self):
        return self.len_

class ZhihuALLData(data.Dataset):

    def __init__(self,train_root,labels_file,type_='char'):
        '''
        Dataset('/mnt/7/zhihu/ieee_zhihu_cup/train.npz','/mnt/7/zhihu/ieee_zhihu_cup/a.json')
        '''
        import json
        with open(labels_file) as f:
            labels_ = json.load(f)

        # embedding_d = np.load(embedding_root)['vector']
        question_d = np.load(train_root)

            # all_data_title,all_data_content =\
        all_char_title,all_char_content=      question_d['title_char'],question_d['content_char']
            # all_data_title,all_data_content =\
        all_word_title,all_word_content=     question_d['title_word'],question_d['content_word']

        self.train_data = (all_char_title[:-200000],all_char_content[:-200000]),( all_word_title[:-200000],all_word_content[:-200000])
        self.val_data = (all_char_title[-200000:],all_char_content[-200000:]), (all_word_title[-200000:],all_word_content[-200000:])
        self.all_num = len(all_char_title)
        # del all_data_title,all_data_content
        
        self.data_title,self.data_content = self.train_data
        self.len_ = len(self.data_title[0])

        self.index2qid = question_d['index2qid'].item()
        self.l_end=0
        self.labels = labels_['d']


    def train(self, train=True):
        if train:
            self.data_title,self.data_content = self.train_data
            self.l_end = 0
        else:
            self.data_title,self.data_content = self.val_data
            self.l_end = self.all_num-200000
        self.len_ = len(self.data_content[0])
        return self

    def __getitem__(self,index):
        '''
        for (title,content),label in dataloader:
            train
`       
        当使用char时
        title: (50,)
        content: (250,)
        labels：(1999,)
        '''
        char,word =  (self.data_title[0][index],self.data_title[1][index]),  (self.data_content[0][index],self.data_content[1][index])
        qid = self.index2qid[index+self.l_end]
        labels = self.labels[qid]
        data = ((t.from_numpy(char[0]).long(),t.from_numpy(char[1]).long()),(t.from_numpy(word[0]).long(),t.from_numpy(word[1]).long()))
        label_tensor = t.zeros(1999).scatter_(0,t.LongTensor(labels),1).long()
        return data,label_tensor

    def __len__(self):
        return self.len_
if __name__=="__main__":
    data_root="/data/text/zhihu/result/"
    labels_file="/home/a/code/pytorch/zhihu/ddd/labels.json"
    val_label="/home/a/code/pytorch/zhihu/ddd/val.npz"
    sb=StackData(data_root,labels_file,val_label)
    for i in range(10):
        print sb[i][0].size()
        print sb[i][1]




















