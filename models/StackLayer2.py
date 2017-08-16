#coding:utf8
from .BasicModule import BasicModule
import torch as t
import numpy as np
from torch import nn
import models
from torch.autograd import Variable
from config import Config
class StackLayer2(BasicModule):
    def __init__(self, opt ):
        super(StackLayer2, self).__init__()
        self.model_name = 'StackLayer2'
        self.opt=opt
        #self.fc=nn.Sequential(
        #    nn.Linear(opt.model_num*opt.num_classes,opt.linear_hidden_size),
        #    nn.BatchNorm1d(opt.linear_hidden_size),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(opt.linear_hidden_size,opt.num_classes)
        #)
        # self.weights = nn.Parameter(t.zeros(opt.num_classes,opt.model_num))
        self.weights=nn.Parameter(t.ones(opt.model_num)/opt.model_num)   
        #self.fc=nn.Linear(opt.model_num*opt.num_classes,opt.num_classes)
        #weights=np.zeros((opt.num_classes,opt.model_num*opt.num_classes),dtype=np.float32)
        #for i in range(opt.model_num):
        #    weights[range(1999),range(i*1999,i*1999+1999)]=0.125
        #self.fc.weight.data=t.from_numpy(weights)
    def forward(self,x):
        # weights = t.nn.functional.softmax(self.weights)
        weights=self.weights/(self.weights.sum()).view(-1).expand_as(self.weights)
        outs =[]
        for i in range(self.opt.model_num):
            ins=x[:,i*self.opt.num_classes:(i+1)*self.opt.num_classes]
            tmp=weights[i].contiguous().view(1,-1).expand_as(ins)*ins
            outs.append(tmp)
        return sum(outs) 
        #x_=x.resize(x.size(0),self.opt.model_num,self.opt.num_classes)
        #return x_.mean(dim=1).view(x.size(0),self.opt.num_classes)
        #return self.fc(x)
    def get_optimizer(self,lr,weight_decay = 0): 
        new_params = [self.weights]
        optimizer=t.optim.Adam([dict(params=new_params,weight_decay = weight_decay,lr=lr),])
        #optimizer = t.optim.Adam(self.weights, lr = lr, weight_decay=weight_decay)
        return optimizer