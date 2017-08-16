from .BasicModule import BasicModule
import torch as t
import numpy as np
from torch import nn


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)

class LSTMText(BasicModule): 
    def __init__(self, opt ):
        super(LSTMText, self).__init__()
        self.model_name = 'LSTMText'
        self.opt=opt

        kernel_size = opt.kernel_size
        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)

        self.title_lstm = nn.LSTM(input_size = opt.embedding_dim,\
                            hidden_size = opt.hidden_size,
                            num_layers = opt.num_layers,
                            bias = True,
                            batch_first = False,
                            # dropout = 0.5,
                            bidirectional = True
                            )
        self.content_lstm =nn.LSTM(  input_size = opt.embedding_dim,\
                            hidden_size = opt.hidden_size,
                            num_layers = opt.num_layers,
                            bias = True,
                            batch_first = False,
                            # dropout = 0.5,
                            bidirectional = True
                            )

        # self.dropout = nn.Dropout()
        self.fc = nn.Sequential(
            nn.Linear(opt.kmax_pooling*(opt.hidden_size*2*2),opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )
        # self.fc = nn.Linear(3 * (opt.title_dim+opt.content_dim), opt.num_classes)
        if opt.embedding_path:
            self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))
 
    def forward(self, title, content):
        title = self.encoder(title)
        content = self.encoder(content)
        if self.opt.static:
            title=title.detach()
            content=content.detach()
        
        title_out = self.title_lstm(title.permute(1,0,2))[0].permute(1,2,0) 

        content_out = self.content_lstm(content.permute(1,0,2))[0].permute(1,2,0)


        title_conv_out = kmax_pooling((title_out),2,self.opt.kmax_pooling)
        content_conv_out = kmax_pooling((content_out),2,self.opt.kmax_pooling)

        conv_out = t.cat((title_conv_out,content_conv_out),dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits

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
