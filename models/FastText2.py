from .BasicModule import BasicModule
import torch as t
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
class FastText2(BasicModule):
    def __init__(self, opt ):
        super(FastText2, self).__init__()
        self.model_name = 'FastText2'
        self.opt=opt
        # self.pre = nn.Sequential(
        #     nn.Linear(opt.embedding_dim,opt.embedding_dim),
        #     nn.BatchNorm1d(opt.embedding_dim),
        #     # nn.ReLU(True)
        # )
        self.pre_fc = nn.Linear(opt.embedding_dim,opt.embedding_dim*2)
        self.bn = nn.BatchNorm1d(opt.embedding_dim*2)
        self.pre_fc2 = nn.Linear(opt.embedding_dim,opt.embedding_dim*2)
        self.bn2 = nn.BatchNorm1d(opt.embedding_dim*2) 
        
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
        
        title_2 = t.nn.functional.relu(self.bn(self.pre_fc(title_em.view(-1,256)).view(title_em.size(0),title_em.size(1),-1).transpose(1,2).contiguous()))
        content_2 = t.nn.functional.relu(self.bn2(self.pre_fc2(content_em.view(-1,256)).view(content_em.size(0),content_em.size(1),-1).transpose(1,2)).contiguous())
    
        # title_2 = self.pre(title_em.contiguous().view(-1,256)).view(title_size)
        # content_2 = self.pre(content_em.contiguous().view(-1,256)).view(content_size)


        title_ = t.mean(title_2,dim=2)
        content_ = t.mean(content_2,dim=2)
        inputs=t.cat((title_.squeeze(),content_.squeeze()),1)
        out=self.fc(inputs.view(inputs.size(0),-1))
        # content_out=self.content_fc(content.view(content.size(0),-1))
        # out=torch.cat((title_out,content_out),1)
        # out=self.fc(out)
        return out
        
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
 