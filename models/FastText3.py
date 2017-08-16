from .BasicModule import BasicModule
import torch as t
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
class FastText3(BasicModule):
    def __init__(self, opt ):
        super(FastText3, self).__init__()
        self.model_name = 'FastText3'
        self.opt=opt
        self.pre1 = nn.Sequential(
            nn.Linear(opt.embedding_dim,opt.embedding_dim*2),
            nn.BatchNorm1d(opt.embedding_dim*2),
            nn.ReLU(True)
        )

        self.pre2 = nn.Sequential(
            nn.Linear(opt.embedding_dim,opt.embedding_dim*2),
            nn.BatchNorm1d(opt.embedding_dim*2),
            nn.ReLU(True)
        )
        # self.pre_fc = nn.Linear(opt.embedding_dim,opt.embedding_dim*2)
        # self.bn = nn.BatchNorm1d(opt.embedding_dim*2)
        # self.pre_fc2 = nn.Linear(opt.embedding_dim,opt.embedding_dim*2)
        # self.bn2 = nn.BatchNorm1d(opt.embedding_dim*2) 

        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(opt.embedding_dim*4,opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )
        if opt.embedding_path:
            print('load embedding')
            self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))
 
    def forward(self,title,content):
        title_em = self.encoder(title)
        content_em = self.encoder(content)
        title_size = title_em.size()
        content_size = content_em.size()
        
 
    
        title_2 = self.pre1(title_em.contiguous().view(-1,256)).view(title_size[0],title_size[1],-1)
        content_2 = self.pre2(content_em.contiguous().view(-1,256)).view(content_size[0],content_size[1],-1)


        title_ = t.mean(title_2,dim=1)
        content_ = t.mean(content_2,dim=1)
        inputs=t.cat((title_.squeeze(),content_.squeeze()),1)
        out=self.fc(inputs)
        # content_out=self.content_fc(content.view(content.size(0),-1))
        # out=torch.cat((title_out,content_out),1)
        # out=self.fc(out)
        return out
        
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
 