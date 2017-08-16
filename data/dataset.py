#encoding:utf-8
from torch.utils import data
import torch as t
import numpy as np
import random
from glob import glob
import json
class StackData(data.Dataset):
    def __init__(self,data_root,labels_file,val):
        '''
        data_root="/data/text/zhihu/result/"
        labels_file="/home/a/code/pytorch/zhihu/ddd/labels.json"
        val="/home/a/code/pytorch/zhihu/ddd/val.npz"
        '''
        self.data_files_path=glob(data_root+"*val.pth")
        print "model num: ",len(self.data_files_path)
        self.model_num=len(self.data_files_path)
        self.all_num=200000
        self.val_num=20000
        with open(labels_file) as f:
            self.label_file= json.load(f)['d']
        self.val=np.load(val)['index2qid'].item()
        self.train_data=t.zeros(self.all_num-self.val_num,1999*self.model_num)
        self.val_data=t.zeros(self.val_num,1999*self.model_num)
        for i in range(self.model_num):
            tmpdata=t.load(self.data_files_path[i]).float()
            self.train_data[:,i*1999:i*1999+1999]=tmpdata[:self.all_num-self.val_num]
            self.val_data[:,i*1999:i*1999+1999]=tmpdata[-1*self.val_num:]
        self.data=self.train_data
        self.len_=self.train_data.size(0)
        self.len_end=0
    def train(self, train=True):
        if train:
            self.data = self.train_data
            self.len_end = 0
        else:
            self.data = self.val_data
            self.len_end = self.all_num-self.val_num
        self.len_ = len(self.data)
        return self
    def __getitem__(self,index):
        qid=self.val[index+self.len_end+2999967-200000]
        data=self.data[index]
        label=self.label_file[qid]
        data = data.float()
        label_tensor = t.zeros(1999).scatter_(0,t.LongTensor(label),1).long()
        return data,label_tensor
    def __len__(self):
        return self.len_  
    
    
class ZhihuData(data.Dataset):
    '''
    主要用到的数据集
    '''

    def __init__(self,train_root,labels_file,type_='char',augument=True):
        '''
        Dataset('/mnt/7/zhihu/ieee_zhihu_cup/train.npz','/mnt/7/zhihu/ieee_zhihu_cup/a.json')
        '''
        import json
        with open(labels_file) as f:
            labels_ = json.load(f)

        # embedding_d = np.load(embedding_root)['vector']
        self.augument=augument
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

        self.index2qid = question_d['index2qid'].item()
        self.l_end=0
        self.labels = labels_['d']
        
        self.training=True

   
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
        title,content =  self.data_title[index],self.data_content[index]
    
        if self.training and self.augument :
            augument=random.random()

            if augument>0.5:
                title = self.dropout(title,p=0.3)
                content = self.dropout(content,p=0.7)
            else:
                title = self.shuffle(title)
                content = self.shuffle(content)

        qid = self.index2qid[index+self.l_end]
        labels = self.labels[qid]
        data = (t.from_numpy(title).long(),t.from_numpy(content).long())
        label_tensor = t.zeros(1999).scatter_(0,t.LongTensor(labels),1).long()
        return data,label_tensor

    def __len__(self):
        return self.len_

class ZhihuALLData(data.Dataset):
    '''
    同时返回word和char的数据
    '''
    def __init__(self,train_root,labels_file,type_='char',augument=True):
        self.augument=augument
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

        self.training=True


    def train(self, train=True):
        if train:
            self.training=True
            self.data_title,self.data_content = self.train_data
            self.l_end = 0
        else:
            self.training=False
            self.data_title,self.data_content = self.val_data
            self.l_end = self.all_num-200000
        self.len_ = len(self.data_content[0])
        return self
    
    def shuffle(self,d):
        return np.random.permutation(d.tolist())

    def dropout(self,d,p=0.5):
        len_ = len(d)
        index = np.random.choice(len_,int(len_*p))
        d[index]=0
        return d   

    def __getitem__(self,index):
        char,word =  (self.data_title[0][index],self.data_title[1][index]),  (self.data_content[0][index],self.data_content[1][index])

        if self.training and self.augument :
            augument=random.random()

            if augument>0.5:
                char = (self.dropout(char[0],p=0.3), self.dropout(char[1],p=0.7))
                word =  (self.dropout(word[0],p=0.3), self.dropout(word[1],p=0.7))
            else:
                char = (self.shuffle(char[0]), self.shuffle(char[1]))
                word =  (self.shuffle(word[0]), self.shuffle(word[1]))
                # title = self.shuffle(title)
                # content = self.shuffle(content)

        qid = self.index2qid[index+self.l_end]
        labels = self.labels[qid]
        data = ((t.from_numpy(char[0]).long(),t.from_numpy(char[1]).long()),(t.from_numpy(word[0]).long(),t.from_numpy(word[1]).long()))
        label_tensor = t.zeros(1999).scatter_(0,t.LongTensor(labels),1).long()
        return data,label_tensor

    def __len__(self):
        return self.len_


class ALLFoldData(data.Dataset):
    '''
    没什么用
    '''

    def __init__(self,train_root,labels_file,type_='char',fold=0):
        '''
        Dataset('/mnt/7/zhihu/ieee_zhihu_cup/train.npz','/mnt/7/zhihu/ieee_zhihu_cup/a.json')
        '''
        import json
        with open(labels_file) as f:
            labels_ = json.load(f)
        self.fold=fold 
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
        self.training=True
        self.index2qid = question_d['index2qid'].item()
        self.l_end=0
        self.labels = labels_['d']


    def train(self, train=True):
        if train:
            self.training=True
            self.data_title,self.data_content = self.train_data
            self.l_end = 0
        else:
            self.training=False
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
        if self.training:
            index = int(index/2)*2+self.fold
            index = index%self.len_
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




















