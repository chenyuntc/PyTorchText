import torch as t
def get_optimizer(model,lr1,lr2=0,weight_decay = 0):
    ignored_params = list(map(id, model.encoder.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                     model.parameters())
    if lr2 is None: lr2 = lr1*0.5 
    optimizer = t.optim.Adam([
            dict(params=base_params,weight_decay = weight_decay,lr=lr1),
            {'params': model.encoder.parameters(), 'lr': lr2}
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

