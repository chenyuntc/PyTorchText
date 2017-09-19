#coding:utf8
from .BasicModule import BasicModule
import torch as t
import numpy as np
from torch import nn
import models
from config import Config 

class MultiModelAll4zhihu(BasicModule): 
    def __init__(self, opt ):
        super(MultiModelAll4zhihu, self).__init__()
        self.model_name = 'MultiModelAll4zhihu'
        self.opt=opt
        self.models = []
        self.word_embedding=nn.Embedding(411720,256)
        self.char_embedding=nn.Embedding(11973,256)
        model_opts = t.load(opt.model_path+'.json')

        for _name,_path,model_opt_ in zip(opt.model_names, opt.model_paths,model_opts):
            tmp_config = Config().parse(model_opt_,print_=False)
            tmp_config.embedding_path=None
            _model = getattr(models,_name)(tmp_config)
            _model.encoder=(self.char_embedding if _model.opt.type_=='char' else self.word_embedding)
            self.models.append(_model)
            
        self.models = nn.ModuleList(self.models)
        self.model_num = len(self.models)
        self.weights = nn.Parameter(t.ones(opt.num_classes,self.model_num))
        self.load(opt.model_path)


    def load(self,path,**kwargs):
        self.load_state_dict(t.load(path)['d'])
    def forward(self, char, word):
        weights = t.nn.functional.softmax(self.weights)
        outs =[]
        for ii,model in enumerate(self.models):
            if model.opt.type_=='char':
                out = t.sigmoid(model(*char))
            else:
                out=t.sigmoid(model(*word))

            out = out*(weights[:,ii].contiguous().view(1,-1).expand_as(out))
            outs.append(out)

        return sum(outs)

 
    def get_optimizer(self,lr1=1e-3,lr2=3e-4,lr3=3e-4,weight_decay = 0):
        encoders = list(self.char_embedding.parameters())+list(self.word_embedding.parameters())
        other_params = [ param_ for model_ in self.models 
                                for name_,param_ in model_.named_parameters()
                                if name_.find('encoder')==-1] 

        new_params = [self.weights]

        optimizer = t.optim.Adam([
                dict(params=other_params,weight_decay = weight_decay,lr=lr1),# conv 全连接 
                dict(params=encoders,weight_decay = weight_decay,lr=lr2), # embedding
                dict(params=new_params,weight_decay = weight_decay,lr=lr3), # 权重
            ])
        return optimizer
 


 
if __name__ == '__main__':
    from ..config import opt
    m = CNNText(opt)
    title = t.autograd.Variable(t.arange(0,500).view(10,50)).long()
    content = t.autograd.Variable(t.arange(0,2500).view(10,250)).long()
    o = m(title,content)
    print(o.size())
