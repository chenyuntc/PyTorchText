#coding:utf8
from config import opt
import models
import os
import tqdm
from data.dataset import ZhihuData
import torch as t
import time
import fire
import torchnet as tnt
from torch.utils import data
from torch.autograd import Variable
from utils.visualize import Visualizer
from utils import get_score
vis = Visualizer(opt.env)

def hook():pass

def val(model,dataset):
    '''
    计算模型在验证集上的分数
    '''

    dataset.train(False)
    model.eval()

    dataloader = data.DataLoader(dataset,
                    batch_size = opt.batch_size,
                    shuffle = False,
                    num_workers = opt.num_workers,
                    pin_memory = True
                    )
    
    predict_label_and_marked_label_list=[]
    for ii,((title,content),label) in tqdm.tqdm(enumerate(dataloader)):
        title,content,label = Variable(title.cuda(),volatile=True),\
                              Variable(content.cuda(),volatile=True),\
                              Variable(label.cuda(),volatile=True)
        score = model(title,content)
        # !TODO: 优化此处代码
        #       1. append
        #       2. for循环
        #       3. topk 代替sort

        predict = score.data.topk(5,dim=1)[1].cpu().tolist()
        true_target = label.data.float().topk(5,dim=1)
        true_index=true_target[1][:,:5]
        true_label=true_target[0][:,:5]
        tmp= []

        for jj in range(label.size(0)):
            true_index_=true_index[jj]
            true_label_=true_label[jj]
            true=true_index_[true_label_>0]
            tmp.append((predict[jj],true.tolist()))
        
        predict_label_and_marked_label_list.extend(tmp)
    del score

    dataset.train(True)
    model.train()
    
    scores,prec_,recall_,_ss=get_score(predict_label_and_marked_label_list)
    return (scores,prec_,recall_,_ss)

def main(**kwargs):
    '''
    训练入口
    '''

    opt.parse(kwargs,print_=False)
    if opt.debug:import ipdb;ipdb.set_trace()

    model = getattr(models,opt.model)(opt).cuda()
    if opt.model_path:
        model.load(opt.model_path)
    print(model)

    opt.parse(kwargs,print_=True)

    vis.reinit(opt.env)
    pre_loss=1.0
    lr,lr2=opt.lr,opt.lr2
    loss_function = getattr(models,opt.loss)()  

    dataset = ZhihuData(opt.train_data_path,opt.labels_path,type_=opt.type_,augument=opt.augument)
    dataloader = data.DataLoader(dataset,
                    batch_size = opt.batch_size,
                    shuffle = opt.shuffle,
                    num_workers = opt.num_workers,
                    pin_memory = True
                    )

    optimizer = model.get_optimizer(lr,opt.lr2,opt.weight_decay)
    loss_meter = tnt.meter.AverageValueMeter()
    score_meter=tnt.meter.AverageValueMeter()
    best_score = 0

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        score_meter.reset()
        for ii,((title,content),label) in tqdm.tqdm(enumerate(dataloader)):
            # 训练 更新参数
            title,content,label = Variable(title.cuda()),Variable(content.cuda()),Variable(label.cuda())
            optimizer.zero_grad()
            score = model(title,content)
            loss = loss_function(score,opt.weight*label.float())
            loss_meter.add(loss.data[0])
            loss.backward()
            optimizer.step()

            if ii%opt.plot_every ==opt.plot_every-1:
                ### 可视化
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

                predict = score.data.topk(5,dim=1)[1].cpu().tolist()
                true_target = label.data.float().cpu().topk(5,dim=1)
                true_index=true_target[1][:,:5]
                true_label=true_target[0][:,:5]
                predict_label_and_marked_label_list=[]
                for jj in range(label.size(0)):
                    true_index_=true_index[jj]
                    true_label_=true_label[jj]
                    true=true_index_[true_label_>0]
                    predict_label_and_marked_label_list.append((predict[jj],true.tolist()))
                score_,prec_,recall_,_ss=get_score(predict_label_and_marked_label_list)
                score_meter.add(score_)
                vis.vis.text('prec:%s,recall:%s,score:%s,a:%s' %(prec_,recall_,score_,_ss),win='tmp')
                vis.plot('scores', score_meter.value()[0])
                
                #eval()
                vis.plot('loss', loss_meter.value()[0])
                k = t.randperm(label.size(0))[0]

            if ii%opt.decay_every == opt.decay_every-1:   
                # 计算在验证集上的分数,并相对应的调整学习率
                del loss
                scores,prec_,recall_ ,_ss= val(model,dataset)
                vis.log({' epoch:':epoch,' lr: ':lr,'scores':scores,'prec':prec_,'recall':recall_,'ss':_ss,'scores_train':score_meter.value()[0],'loss':loss_meter.value()[0]})
                
                if scores>best_score:
                    best_score = scores
                    best_path = model.save(name = str(scores),new=True)
               
                if scores < best_score:
                    model.load(best_path,change_opt=False)
                    lr = lr * opt.lr_decay
                    lr2= 2e-4 if lr2==0 else  lr2*0.8
                    optimizer = model.get_optimizer(lr,lr2,0)                        
                
                pre_loss = loss_meter.value()[0]
                loss_meter.reset()
                score_meter.reset()

if __name__=="__main__":
    fire.Fire()  
