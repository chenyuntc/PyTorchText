#coding:utf8
from .BasicModule import BasicModule
import torch as t
import numpy as np
from torch import nn
import models
from config import Config 

class BoostModel2(BasicModule): 
    def __init__(self, opt ):
        super(BoostModel2, self).__init__()
        self.model_name = 'BoostModel2'
        self.opt=opt
        # self.char_models = []
        self.models = []
        new_model=[]
        self.word_embedding=nn.Embedding(411720,256)
        self.char_embedding=nn.Embedding(11973,256)
        if opt.embedding_path:
            self.word_embedding.weight.data.copy_(t.from_numpy(np.load('ddd/word_embedding_new.npz')['vector']))
            self.char_embedding.weight.data.copy_(t.from_numpy(np.load('ddd/char_embedding_new.npz')['vector']))

        for _name,_path in zip(opt.model_names, opt.model_paths):
            _names = _name.split('-')
            _name = _names[0]
            tmp_config = Config().parse(opt.state_dict(),print_=False)
            tmp_config.embedding_path=None
            _model = getattr(models,_name)(tmp_config)
            if _path is not None:
                _model.load(_path)
        
            _model.encoder=(self.char_embedding if _model.opt.type_=='char' else self.word_embedding)
            if len(_names)>1:new_model.append(_model)
            else: 
                self.models.append(_model)
        self.models=nn.ModuleList(self.models)
        model_num=len(self.models) 
        if opt.model_path:
            state_dict = t.load(opt.model_path)['d']
            for key in state_dict:
                if key.startswith('new_model'):
                    key2=key.replace('new_model','models.%s' %(model_num-1))
                    state_dict[key2] = state_dict[key]
                    del state_dict[key]
            self.load_state_dict(state_dict)
        
        self.new_model = nn.ModuleList(new_model)


        self.model_num = len(self.models)
        assert self.opt.loss=='bceloss'

    def load(self,path,**kwargs):
        self.load_state_dict(t.load(path)['d'])

    def forward(self, char, word):
        self.models.eval()
        outs =[]
        for ii,model in enumerate(self.models):
            if model.opt.type_=='char':
                out = t.sigmoid(model(*char))
            else:
                out=t.sigmoid(model(*word))
            outs.append(out.detach())
        for ii,model in enumerate(self.new_model):
            if model.opt.type_=='char':
                out = t.sigmoid(model(*char))
            else:
                out=t.sigmoid(model(*word))
            outs.append(out)
        return sum(outs)/(len(outs))

 
    def get_optimizer(self,lr1=0,lr2=1e-4,lr3=1e-3,weight_decay = 0):
        encoders = list(self.char_embedding.parameters())+list(self.word_embedding.parameters())
        other_params = [ param_ for model_ in self.models 
                                for name_,param_ in model_.named_parameters()
                                if name_.find('encoder')==-1] 

        new_params = [param_ for name_,param_ in self.new_model.named_parameters() if name_.find('encoder')==-1]

        optimizer = t.optim.Adam([
                # dict(params=other_params,weight_decay = weight_decay,lr=lr1),# conv 全连接 
                dict(params=encoders,weight_decay = weight_decay,lr=lr2), # embedding
                dict(params=new_params,weight_decay = weight_decay,lr=lr1), # 新增加的子网络
            ])
        return optimizer
 


 
if __name__ == '__main__':
    from ..config import opt
    m = CNNText(opt)
    title = t.autograd.Variable(t.arange(0,500).view(10,50)).long()
    content = t.autograd.Variable(t.arange(0,2500).view(10,250)).long()
    o = m(title,content)
    print(o.size())
