#coding:utf8
from .BasicModule import BasicModule
import torch as t
import numpy as np
from torch import nn
import models
from config import Config 



def kmax_pooling(x, dim=2, k=1):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class FastText(nn.Module):
    def __init__(self,opt):
        super(FastText, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(opt.embedding_dim*2,opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )
    def forward(self,title,content):
        return self.fc(t.cat((title.mean(1),content.mean(1)),dim=1).view(title.size(0),-1))
class RNN(nn.Module):
    def __init__(self,opt):
        super(RNN, self).__init__()
        if opt.type_=='word':pass
        self.lstm = nn.LSTM(input_size = opt.embedding_dim,\
                            hidden_size = opt.hidden_size,
                            num_layers = opt.num_layers,
                            bias = True,
                            batch_first = False,
                            # dropout = 0.5,
                            bidirectional = True
                            )
    
        self.fc = nn.Sequential(
            nn.Linear((opt.hidden_size*2*2),opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )

    def forward(self,title,content):
        title_lstm = self.lstm(title.permute(1,0,2))[0].permute(1,2,0) 
        content_lstm = self.lstm(content.permute(1,0,2))[0].permute(1,2,0) 
        lstm_out = t.cat((kmax_pooling(title_lstm,k=1),kmax_pooling(content_lstm,k=1)),dim=1)
        reshaped = lstm_out.view(lstm_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits,(title_lstm,content_lstm)



class CNN(nn.Module):
    def __init__(self,opt):
        super(CNN, self).__init__()
        if opt.type_=='word':
            kernel_sizes1=[1,2,3,4,4]
            kernel_sizes2=[1,2,2,2,3]
        else:
            kernel_sizes1=[2,3,5,6,8]
            kernel_sizes2=[1,2,3,3,4]
        self.convs=[ nn.Sequential(
                                nn.Conv1d(in_channels = opt.embedding_dim,
                                        out_channels = opt.title_dim,
                                        kernel_size = kernel_size1),
                                nn.BatchNorm1d(opt.title_dim),
                                nn.ReLU(inplace=True),

                                nn.Conv1d(in_channels = opt.title_dim,
                                out_channels = opt.title_dim,
                                kernel_size = kernel_size2),
                                nn.BatchNorm1d(opt.title_dim),
                                nn.ReLU(inplace=True),
                            )
         for kernel_size1,kernel_size2  in zip(kernel_sizes1,kernel_sizes2)]
        self.convs=nn.ModuleList(self.convs)

        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes1)*(opt.title_dim*2),opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )

    def forward(self,title,content):
         title_outs = [kmax_pooling(model_(title.permute(0, 2, 1))) for model_ in self.convs]
         content_outs = [kmax_pooling(model_(content.permute(0, 2, 1))) for model_ in self.convs]
         conv_out = t.cat((title_outs+content_outs),dim=1)
         reshaped = conv_out.view(conv_out.size(0), -1)
         logits = self.fc((reshaped))
         return logits








class RCNN(nn.Module):
    def __init__(self,opt):
        super(RCNN, self).__init__()
        kernel_size = 2 if opt.type_=='word' else 3
        self.conv = nn.Sequential(
                                nn.Conv1d(in_channels = opt.hidden_size*2 + opt.embedding_dim,
                                        out_channels = opt.title_dim*3,
                                        kernel_size = kernel_size),
                                nn.BatchNorm1d(opt.title_dim*3),
                                nn.ReLU(inplace=True),
                                nn.Conv1d(in_channels = opt.title_dim*3,
                                        out_channels = opt.title_dim*3,
                                        kernel_size = kernel_size),
                                nn.BatchNorm1d(opt.title_dim*3),
                                nn.ReLU(inplace=True),
                                # nn.MaxPool1d(kernel_size = (opt.title_seq_len - kernel_size + 1))
                            ) 
        self.fc=nn.Linear((opt.title_dim*3*2),opt.num_classes)
    def forward(self,rnn_out,em):
        title_em,content_em = em
        title,content = rnn_out
        title_out = self.conv(t.cat((title,title_em.permute(0,2,1)),dim=1))
        content_out = self.conv(t.cat((content,content_em.permute(0,2,1)),dim=1))
        conv_out = t.cat((kmax_pooling(title_out),kmax_pooling(content_out)),dim=1) 
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits








###
#4#
###
class FourModel(BasicModule): 

    def __init__(self, opt ):
        super(FourModel, self).__init__()
        self.model_name = 'FourModel'
        self.opt=opt
        # self.char_models = []
        
        self.word_embedding=nn.Embedding(411720,256)
        self.char_embedding=nn.Embedding(11973,256)
        if opt.embedding_path:
            self.word_embedding.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path.replace('char','word'))['vector']))
            self.char_embedding.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path.replace('word','char'))['vector']))

        self.fasttext = FastText(opt)
        self.cnn = CNN(opt) 
        self.rnn =RNN(opt)
        self.rcnn = RCNN(opt)

        # opt.type_='char'
        opt = opt.parse(dict(type_='char'),print_=False)
        self.fasttext2 = FastText(opt)
        self.cnn2 = CNN(opt) 
        self.rnn2 =RNN(opt)
        self.RCNN2 = RCNN(opt)

        opt.type_='word'
    

        self.model_num = 8
        self.weights = nn.Parameter(t.ones(opt.num_classes,self.model_num))
        assert self.opt.loss=='bceloss'


    def forward(self, char, word):
        char_em = self.char_embedding(char[0]),self.char_embedding(char[1])
        word_em = self.word_embedding(word[0]),self.word_embedding(word[1])
        weights = t.nn.functional.softmax(self.weights)
        outs =[]
        fasttext = t.sigmoid(self.fasttext(*word_em))
        cnn = t.sigmoid(self.cnn(*word_em))
        rnn,rnn_out = (self.rnn(*word_em))
        rnn = t.sigmoid(rnn)
        rcnn = t.sigmoid(self.rcnn(rnn_out,word_em))


        fasttext2 = t.sigmoid(self.fasttext(*char_em))
        cnn2 = t.sigmoid(self.cnn(*char_em))
        # rnn2,rnn_out2 = t.sigmoid(self.rnn(*char_em))
        rnn2,rnn_out2 = (self.rnn(*char_em))
        rnn2 = t.sigmoid(rnn2)
        rcnn2 = t.sigmoid(self.rcnn(rnn_out2,char_em))


        results = fasttext*(weights[:,0].contiguous().view(1,-1).expand_as(fasttext)) +\
                  cnn*(weights[:,1].contiguous().view(1,-1).expand_as(cnn)) +\
                  rnn*(weights[:,2].contiguous().view(1,-1).expand_as(rnn)) + \
                  rcnn*(weights[:,3].contiguous().view(1,-1).expand_as(rcnn)) +\
                  fasttext2*(weights[:,4].contiguous().view(1,-1).expand_as(fasttext2)) +\
                  cnn2*(weights[:,5].contiguous().view(1,-1).expand_as(cnn2)) +\
                  rnn2*(weights[:,6].contiguous().view(1,-1).expand_as(rnn2)) + \
                  rcnn2*(weights[:,7].contiguous().view(1,-1).expand_as(rcnn2))


        return results 
        # for ii,model in enumerate(self.models):
        #     if model.opt.type_=='char':
        #         out = t.sigmoid(model(*char))
        #     else:
        #         out=t.sigmoid(model(*word))

        #     out = out*(weights[:,ii].contiguous().view(1,-1).expand_as(out))
        #     outs.append(out)
        #     # outs = [t.sigmoid(model(title,content))*weight  for model in  self.models]

        # # outs = [model(title,content)*weight.view(1,1).expand(title.size(0),self.opt.num_classes).mm(self.label_weight) for model,weight in zip(self.models,self.weight)]
        # return sum(outs)

 
    def get_optimizer(self,lr1=1e-3,lr2=1e-4,lr3=0,weight_decay = 0):
        encoders = list(self.char_embedding.parameters())+list(self.word_embedding.parameters())
        other_params = [ param_
                                for name_,param_ in self.named_parameters()
                                if name_.find('embedding')==-1 and name_.find('weight')==-1] 
        # new_params = [self.weights ,self.label_weight]
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
