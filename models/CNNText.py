from .BasicModule import BasicModule
import torch as t
import numpy as np
from torch import nn

class CNNText(BasicModule): 
    def __init__(self, opt ):
        super(CNNText, self).__init__()
        self.model_name = 'CNNText'
        self.opt=opt
        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)

        self.title_conv = nn.Sequential(
            nn.Conv1d(in_channels = opt.embedding_dim,
                      out_channels = opt.title_dim,
                      kernel_size = opt.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = (opt.title_seq_len - opt.kernel_size + 1))
        )

        self.content_conv = nn.Sequential(
            nn.Conv1d(in_channels = opt.embedding_dim,
                      out_channels = opt.content_dim,
                      kernel_size = opt.kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = (opt.content_seq_len - opt.kernel_size + 1))
        )

        self.fc = nn.Linear(opt.title_dim+opt.content_dim, opt.num_classes)
        if opt.embedding_path:
            self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))
 


    def forward(self, title, content):
        title = self.encoder(title)
        content = self.encoder(content)
        title_out = self.title_conv(title.permute(0, 2, 1))
        content_out = self.content_conv(content.permute(0,2,1))
        conv_out = t.cat((title_out,content_out),dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(reshaped)
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
