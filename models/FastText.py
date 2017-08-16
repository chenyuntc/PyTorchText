from .BasicModule import BasicModule
import torch as t
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
class FastText(BasicModule):
    def __init__(self, opt ):
        super(FastText, self).__init__()
        self.model_name = 'FastText'
        self.opt=opt
        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)
        # self.title_fc = nn.Sequential(#2 fc layer score 0.40039 visdom fasttext_15:17
        #     nn.Linear(opt.embedding_dim*2,opt.linear_hidden_size),
        #     nn.BatchNorm1d(opt.linear_hidden_size),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(opt.linear_hidden_size,opt.num_classes)
        # )
        self.title_fc = nn.Sequential(
            nn.Linear(opt.embedding_dim,opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            # nn.Linear(opt.linear_hidden_size,opt.linear_hidden_size),
            # nn.BatchNorm1d(opt.linear_hidden_size),
            # nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )
        self.content_fc = nn.Sequential(
            nn.Linear(opt.embedding_dim,opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            # nn.Linear(opt.linear_hidden_size,opt.linear_hidden_size),
            # nn.BatchNorm1d(opt.linear_hidden_size),
            # nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )
        if opt.embedding_path:
            print('load embedding')
            self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))
 
    def forward(self,title,content):
        title_ = t.mean(self.encoder(title),dim=1)
        content_=t.mean(self.encoder(content),dim=1)
        #inputs=torch.cat((title_,content_),2)
        out1=self.title_fc(title_.view(title_.size(0),-1))
        out2=self.content_fc(content_.view(content_.size(0),-1))
        # content_out=self.content_fc(content.view(content.size(0),-1))
        # out=torch.cat((title_out,content_out),1)
        # out=self.fc(out)
        return 0.99*out1+0.01*out2
        
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
 