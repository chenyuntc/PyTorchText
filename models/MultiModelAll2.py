#coding:utf8
from .BasicModule import BasicModule
import torch as t
import numpy as np
from torch import nn
import models
from config import Config 

class MultiModelAll2(BasicModule): 
    def __init__(self, opt ):
        super(MultiModelAll2, self).__init__()
        self.model_name = 'MultiModelAll2'
        self.opt=opt
        self.models = []

        for _name,_path in zip(opt.model_names, opt.model_paths):
            tmp_config = Config().parse(opt.state_dict(),print_=False)
            # tmp_config.static=True
            tmp_config.embedding_path=None
            _model = getattr(models,_name)(tmp_config)
            if _path is not None:
                _model.load(_path)
            self.models.append(_model)

        self.models = nn.ModuleList(self.models)
        self.model_num = len(self.models)
        self.weights = nn.Parameter(t.ones(opt.num_classes,self.model_num))
        assert self.opt.loss=='bceloss'

        self.eval()

    def reinit(self):
        pass


    def forward(self, char, word):
        weights = t.nn.functional.softmax(self.weights)
        outs =[]
        for ii,model in enumerate(self.models):
            if model.opt.type_=='char':
                out = t.sigmoid(model(*char))
            else:
                out=t.sigmoid(model(*word))
            if self.opt.static:     out = out.detach()
            out = out*(weights[:,ii].contiguous().view(1,-1).expand_as(out))
            outs.append(out)
            # outs = [t.sigmoid(model(title,content))*weight  for model in  self.models]

        # outs = [model(title,content)*weight.view(1,1).expand(title.size(0),self.opt.num_classes).mm(self.label_weight) for model,weight in zip(self.models,self.weight)]
        return sum(outs)

 
    def get_optimizer(self,lr1=1e-4,lr2=1e-4,lr3=0,weight_decay = 0):
        # encoders = list(self.char_embedding.parameters())+list(self.word_embedding.parameters())
        other_params = [ param_ for model_ in self.models 
                                for name_,param_ in model_.named_parameters()
                                if name_.find('encoder')==-1] 
        encoders = [ param_ for model_ in self.models 
                                for name_,param_ in model_.named_parameters()
                                if name_.find('encoder')!=-1] 
        new_params = [self.weights]

        optimizer = t.optim.Adam([
                dict(params=other_params,weight_decay = weight_decay,lr=lr1),# conv 全连接 
                dict(params=encoders,weight_decay = weight_decay,lr=lr2), # embedding
                dict(params=new_params,weight_decay = weight_decay,lr=lr3), # 权重
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
