import numpy as np

import torch as t
from torch import nn

from .BasicModule import BasicModule


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class RNNText2(BasicModule):
    def __init__(self, opt):
        super(RNNText2, self).__init__()
        self.model_name = 'RNNText2'
        self.opt = opt

        # kernel_size = opt.kernel_size
        self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)

        self.title_lstm = nn.LSTM(input_size=opt.embedding_dim,
                                  hidden_size=opt.hidden_size,
                                  num_layers=1,
                                  bias=True,
                                  batch_first=False,
                                #   dropout=0.5,
                                  bidirectional=True
                                  )
        self.title_conv = nn.ModuleList([nn.Sequential(
                        nn.Conv1d(in_channels=opt.hidden_size * 2 + opt.embedding_dim,
                                out_channels=opt.title_dim,
                                kernel_size=kernel_size),
                        nn.ReLU(),
                        nn.MaxPool1d(kernel_size=(opt.title_seq_len - kernel_size + 1))
                    ) for kernel_size in opt.kernel_sizes])

        self.content_lstm = nn.LSTM(input_size=opt.embedding_dim,
                                    hidden_size=opt.hidden_size,
                                    num_layers=1,
                                    bias=True,
                                    batch_first=False,
                                    # dropout=0.5,
                                    bidirectional=True
                                    )

        self.content_conv = nn.ModuleList([nn.Sequential(
                                nn.Conv1d(in_channels=opt.hidden_size * 2 + opt.embedding_dim,
                                        out_channels=opt.content_dim,
                                        kernel_size=kernel_size),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size=(opt.content_seq_len - kernel_size + 1))
                            )
                                for kernel_size in opt.kernel_sizes  ])

        self.dropout = nn.Dropout()
        self.fc = nn.Linear(len(opt.kernel_sizes) *
                            (opt.title_dim + opt.content_dim), opt.num_classes)

        if opt.embedding_path:
            self.encoder.weight.data.copy_(t.from_numpy(
                np.load(opt.embedding_path)['vector']))

    def forward(self, title, content):
        title = self.encoder(title)
        content = self.encoder(content)

        title_out = self.title_lstm(title.permute(1, 0, 2))[0].permute(1, 2, 0)
        title_em = title.permute(0, 2, 1)
        title_out = t.cat((title_out, title_em), dim=1)

        content_out = self.content_lstm(content.permute(1, 0, 2))[
            0].permute(1, 2, 0)
        content_em = (content).permute(0, 2, 1)
        content_out = t.cat((content_out, content_em), dim=1)

        title_conv_out = [m((title_out)) for m in self.title_conv]
        content_conv_out = [m((content_out)) for m in self.content_conv]

        conv_out = t.cat((title_conv_out + content_conv_out), dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(self.dropout(reshaped))
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
    # from ..config import opt
    # m = CNNText(opt)
    # title = t.autograd.Variable(t.arange(0, 500).view(10, 50)).long()
    # content = t.autograd.Variable(t.arange(0, 2500).view(10, 250)).long()
    # o = m(title, content)
    # print(o.size())
    pass
