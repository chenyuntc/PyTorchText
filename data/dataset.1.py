#encoding:utf-8
from torch.utils import data
import torch as t
import numpy as np
import random
from glob import glob
class StackData(data.Dataset):
    def __init__(self,data_root,labels_file):
        self.data_files_path=glob(data_root+"*val.pth")
        self.model_num=len(self.data_files_path)
        self.label_file_path=labels_file
        self.data=t.zeros(100,1999*self.model_num)
        for i in range(self.model_num):
            self.data[:,i*1999:i*1999+1999]=t.sigmoid(t.load(self.data_files_path[i]).float()[:100]) 
        print self.data.size()
        
class ZhihuData(data.Dataset):

    def __init__(self,train_root,labels_file,type_='char'):
        '''
        Dataset('/mnt/7/zhihu/ieee_zhihu_cup/train.npz','/mnt/7/zhihu/ieee_zhihu_cup/a.json')
        '''
        import json
        with open(labels_file) as f:
            labels_ = json.load(f)

        # embedding_d = np.load(embedding_root)['vector']
        question_d = np.load(train_root)
        self.type_=type_
        if type_ == 'char':
            all_data_title,all_data_content =\
                 question_d['title_char'],question_d['content_char']

        elif type_ == 'word':
            all_data_title,all_data_content =\
                 question_d['title_word'],question_d['content_word']

        self.train_data = all_data_title[:-20000],all_data_content[:-20000]
        self.val_data = all_data_title[-20000:],all_data_content[-20000:]

        self.all_num = len(all_data_content)
        # del all_data_title,all_data_content
        
        self.data_title,self.data_content = self.train_data
        self.len_ = len(self.data_title)

        self.index2qid = question_d['index2qid'].item()
        self.l_end=0
        self.labels = labels_['d']

    # def augument(self,d):
    #     '''
    #     数据增强之:   随机偏移
    #     '''
    #     if self.type_=='char':
    #         _index = (-8,8)
    #     else :_index =(-5,5)
    #     r = d.new(d.size()).fill_(0)
    #     index = random.randint(-3,4)
    #     if _index >0:
    #         r[index:] = d[:-index]
    #     else:
    #         r[:-index] = d[index:]
    #     return r

    # def augument(self,d,type_=1):
    #     if type_==1:
    #         return self.shuffle(d)
    #     else :
    #         if self.type_=='char':
    #             return self.dropout(d,p=0.6)

    def shuffle(self,d):
        return np.random.permutation(d.tolist())

    def dropout(self,d,p=0.5):
        len_ = len(d)
        index = np.random.choice(len_,int(len_*p))
        d[index]=0
        return d     

    def train(self, train=True):
        if train:
            self.data_title,self.data_content = self.train_data
            self.l_end = 0
        else:
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
        title,content =  self.data_title[index],self.data_content[index]
        qid = self.index2qid[index+self.l_end]
        labels = self.labels[qid]
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

        self.train_data = (all_char_title[:-20000],all_char_content[:-20000]),( all_word_title[:-20000],all_word_content[:-20000])
        self.val_data = (all_char_title[-20000:],all_char_content[-20000:]), (all_word_title[-20000:],all_word_content[-20000:])
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
            self.l_end = self.all_num-20000
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
        char,word =  (self.data_titl