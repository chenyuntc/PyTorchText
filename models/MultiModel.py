#coding:utf8
from .BasicModule import BasicModule
import torch as t
import numpy as np
from torch import nn
import models
from config import Config 

class MultiModel(BasicModule): 
    def __init__(self, opt ):
        super(MultiModel, self).__init__()
        self.model_name = 'MultiModel'
        self.opt=opt
        self.models = []
        for _name,_path in zip(opt.model_names, opt.model_paths):
            _model = getattr(models,_name)(Config().parse(opt.state_dict(),print_=False))
            if _path is not None:
                _model.load(_path)
            self.models.append(_model)
        self.models = nn.ModuleList(self.models)
        self.model_num = len(self.models)
        self.weights = nn.Parameter(t.ones(opt.num_classes,self.model_num))
        # self.weight =[nn.Parameter(t.ones(self.model_num)/self.model_num) for _ in range(self.model_num)]
        # self.label_weight = nn.Parameter(t.eye(opt.num_classes))
       
    def forward(self, title, content):
        weights = t.nn.functional.softmax(self.weights)
        outs =[]
        for ii,model in enumerate(self.models):
            out = t.sigmoid(model(title,content))
            out = out*(weights[:,ii].contiguous().view(1,-1).expand_as(out))
            outs.append(out)
            # outs = [t.sigmoid(model(title,content))*weight  for model in  self.models]

        # outs = [model(title,content)*weight.view(1,1).expand(title.size(0),self.opt.num_classes).mm(self.label_weight) for model,weight in zip(self.models,self.weight)]
        return sum(outs)

 
    def get_optimizer(self,lr1,lr2=0,weight_decay = 0):
        encoders = [param_ for model_ in self.models 
                           for param_ in model_.encoder.parameters()]
        other_params = [ param_ for model_ in self.models 
                                for name_,param_ in model_.named_parameters()
                                if name_.find('encoder')==-1] 
        # new_params = [self.weights ,self.label_weight]
        new_params = [self.weights]

        optimizer = t.optim.Adam([
                dict(params=other_params,weight_decay = weight_decay,lr=lr1),# conv 全连接 
                dict(params=encoders,weight_decay = weight_decay,lr=lr2), # embedding
                dict(params=new_params,weight_decay = weight_decay,lr=lr1), # 权重
            ])
        return optimizer
 
    # def get_optimizer(self):  
    #    return  t.optim.Adam([
    #             {'params': self.title_conv.parameters()},
    #             {'params': self.content_conv.parameters()},
    #             {'params': self.fc.parameters()},
    #             {'params': self.encoder.parameters(), 'lr': 5e-4}
    #         ], lr=self.opt.lr)
    # # end method forward


 
if __name__ == '__main__':
    from ..config import opt
    m = CNNText(opt)
    title = t.autograd.Variable(t.arange(0,500).view(10,50)).long()
    content = t.autograd.Variable(t.arange(0,2500).view(10,250)).long()
    o = m(title,content)
    print(o.size())
