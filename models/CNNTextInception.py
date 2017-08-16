from .BasicModule import BasicModule
import torch as t
import torch
import numpy as np
from torch import nn
from collections import OrderedDict

class Inception(nn.Module):
    def __init__(self,cin,co,relu=True,norm=True):
        super(Inception, self).__init__()
        assert(co%5==0)
        cos=[co/5]*5
        self.activa=nn.Sequential()
        if norm:self.activa.add_module('norm',nn.BatchNorm1d(co))
        if relu:self.activa.add_module('relu',nn.ReLU(True))
        self.branch1 =nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(cin,cos[0], 1,stride=1)),
            ])) 
        self.branch2 =nn.Sequential(OrderedDict([
            ('conv3', nn.Conv1d(cin,cos[1], 2,stride=1,padding=1)),
            ]))
        self.branch3 =nn.Sequential(OrderedDict([
            ('conv3', nn.Conv1d(cin,cos[2], 3,stride=1,padding=1)),
            ]))
        self.branch4 =nn.Sequential(OrderedDict([
            ('conv3', nn.Conv1d(cin,cos[3], 4,stride=1,padding=2)),
            ]))
        self.branch5 =nn.Sequential(OrderedDict([
            ('conv3', nn.Conv1d(cin,cos[4], 5,stride=1,padding=2)),
            ]))
    def forward(self,x):
        branch1=self.branch1(x)
        branch2=self.branch2(x)[:,:,1:]
        branch3=self.branch3(x)
        branch4=self.branch4(x)[:,:,1:]
        branch5=self.branch5(x)
        result=self.activa(torch.cat((branch1,branch2,branch3,branch4,branch5),1))
        return result
class CNNTextInception(BasicModule):
    def __init__(self, opt ):
        super(CNNTextInception, self).__init__()
        incept_dim=opt.inception_dim
        self.model_name = 'CNNTextInception'
        self.opt=opt
        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)
        self.title_conv=nn.Sequential(
            Inception(opt.embedding_dim,incept_dim),#(batch_size,64,opt.title_seq_len)->(batch_size,32,(opt.title_seq_len)/2)
            Inception(incept_dim,incept_dim),
            Inception(incept_dim,incept_dim),
            nn.MaxPool1d(opt.title_seq_len)
        )
        self.content_conv=nn.Sequential(
            Inception(opt.embedding_dim,incept_dim),#(batch_size,64,opt.content_seq_len)->(batch_size,64,(opt.content_seq_len)/2)
            #Inception(incept_dim,incept_dim),#(batch_size,64,opt.content_seq_len/2)->(batch_size,32,(opt.content_seq_len)/4)
            Inception(incept_dim,incept_dim),
            Inception(incept_dim,incept_dim),
            nn.MaxPool1d(opt.content_seq_len)
        )
        self.fc = nn.Sequential(
            nn.Linear(incept_dim*2,opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )
    def forward(self,title,content):
        title = self.encoder(title)
        content=self.encoder(content)
        title_out=self.title_conv(title.permute(0,2,1))
        content_out=self.content_conv(content.permute(0,2,1))
        out=torch.cat((title_out,content_out),1).view(content_out.size(0), -1)
        out=self.fc(out)
        return out
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        