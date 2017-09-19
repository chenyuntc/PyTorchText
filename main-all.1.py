#coding:utf8
from config import opt
import models
import os
import tqdm
from data.dataset import ZhihuData,ZhihuALLData
import torch as t
import time
import fire
import torchnet as tnt
from torch.utils import data
from torch.autograd import Variable
from utils.visualize import Visualizer
from utils import get_score#,get_optimizer 
vis = Visualizer(opt.env)
'''
直接训练最好模型的attention stack
'''
def hook():pass

def val(model,dataset):

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
        title,content,label = (Variable(title[0].cuda()),Variable(title[1].cuda())),(Variable(content[0].cuda()),Variable(content[1].cuda())),Variable(label.cuda())
        score = model(title,content)
        # !TODO: 优化此处代码
        #       1. append
        #       2. for循环
        #       3. topk 代替sort

        predict = score.data.topk(5,dim=1)[1].cpu().tolist()
        true_target = label.data.float().topk(5,dim=1)#[1].cpu().tolist()#sort(dim=1,descending=True)
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
    #动态加全职衰减
    opt.parse(kwargs,print_=False)
    if opt.debug:import ipdb;ipdb.set_trace()
    # opt.model_names=['MultiCNNTextBNDeep','RCNN','LSTMText','CNNText_inception','RCNN','CNNText_inception','LSTMText']
    # opt.model_paths = ['checkpoints/MultiCNNTextBNDeep_word_0.41124002492','checkpoints/RCNN_word_0.411511574999','checkpoints/LSTMText_word_0.411994005382','checkpoints/CNNText_tmp_char_0.402429167301','checkpoints/RCNN_char_0.403710422571','checkpoints/CNNText_tmp_word_0.41096749885','checkpoints/LSTMText_char_0.403192339135',]#'checkpoints/FastText_word_0.400391584867']
##################iMultiModelAll2_word_0.425600838271################
    # opt.model_names=['MultiCNNTextBNDeep','RCNN','LSTMText','RCNN','CNNText_inception']
    # opt.model_paths = ['checkpoints/MultiCNNTextBNDeep_word_0.41124002492','checkpoints/RCNN_word_0.411511574999','checkpoints/LSTMText_word_0.411994005382','checkpoints/RCNN_char_0.403710422571','checkpoints/CNNText_tmp_char_0.402429167301']
#####-------------------------------------------------------#####

#############################################################
    # opt.model_names=['MultiCNNTextBNDeep','RCNN','LSTMText','RCNN','CNNText_inception']
    # opt.model_paths = ['checkpoints/MultiCNNTextBNDeep_0.37125473788','checkpoints/RCNN_word_0.373609030286','checkpoints/LSTMText_word_0.381833388089','checkpoints/RCNN_char_0.3456599248','checkpoints/CNNText_tmp_0.352036505041']
#####-------------------------------------------------------#####

    # opt.model_names=['LSTMText','MultiCNNTextBNDeep']
    # opt.model_paths=['checkpoints/LSTMText_word_0.396765494482','checkpoints/MultiCNNTextBNDeep_word_0.391018392216']
    # opt.fold=1
    # from data.dataset import ALLFoldData as ZhihuALLData
########################################################################
    # opt.model_names=['MultiCNNTextBNDeep','RCNN','LSTMText','RCNN','CNNText_inception']
    # opt.model_paths = ['checkpoints/MultiCNNTextBNDeep_0.37125473788','checkpoints/RCNN_word_0.373609030286','checkpoints/LSTMText_word_0.381833388089','checkpoints/RCNN_char_0.3456599248','checkpoints/CNNText_tmp_0.352036505041']

#######################################0.41884129858126845-force#####################
    # opt.model_names=['MultiCNNTextBNDeep','RCNN','LSTMText','RCNN','MultiCNNTextBNDeep']
    # opt.model_paths = ['checkpoints/MultiCNNTextBNDeep_word_0.410011182415','checkpoints/RCNN_word_0.413446202556','checkpoints/LSTMText_word_0.413681107036','checkpoints/RCNN_char_0.398655349075','checkpoints/MultiCNNTextBNDeep_char_0.38666657051']
#######################################################################

#############################################MultiModelallfast_0.419088#####################################
    opt.model_names=['MultiCNNTextBNDeep','FastText3','LSTMText','CNNText_inception']
    opt.model_paths = ['checkpoints/MultiCNNTextBNDeep_word_0.41124002492','checkpoints/FastText3_word_0.40810787337','checkpoints/LSTMText_word_0.413681107036','checkpoints/CNNText_tmp_char_0.402429167301'
########################################################################################3
    model = getattr(models,opt.model)(opt).cuda()
    if opt.model_path:
        model.load(opt.model_path)
    print(model)

    opt.parse(kwargs,print_=True)

    vis.reinit(opt.env)
    pre_loss=1.0
    lr,lr2=opt.lr,opt.lr2
    loss_function = getattr(models,opt.loss)()  
    if opt.all:dataset = ZhihuALLData(opt.train_data_path,opt.labels_path,type_=opt.type_,augument=opt.augument)
    # else :dataset = ZhihuData(opt.train_data_path,opt.labels_path,type_=opt.type_)
    dataloader = data.DataLoader(dataset,
                    batch_size = opt.batch_size,
                    shuffle = opt.shuffle,
                    num_workers = opt.num_workers,
                    pin_memory = True
                    )

    optimizer = model.get_optimizer(opt.lr,opt.lr2)
    loss_meter = tnt.meter.AverageValueMeter()
    score_meter=tnt.meter.AverageValueMeter()
    best_score = 0
    # pre_score = 0

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        score_meter.reset()
        for ii,((title,content),label) in tqdm.tqdm(enumerate(dataloader)):
            title,content,label = (Variable(title[0].cuda()),Variable(title[1].cuda())),(Variable(content[0].cuda()),Variable(content[1].cuda())),Variable(label.cuda())
            optimizer.zero_grad()
            score = model(title,content)
            loss = loss_function(score,label.float())
            loss_meter.add(loss.data[0])
            loss.backward()
            optimizer.step()

            if ii%opt.plot_every ==opt.plot_every-1:
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

                predict = score.data.topk(5,dim=1)[1].cpu().tolist()#(dim=1,descending=True)[1][:,:5].tolist()
                true_target = label.data.float().topk(5,dim=1)#[1].cpu().tolist()#sort(dim=1,descending=True)
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
                vis.plot('loss', loss_meter.value()[0])
                   
            if ii%opt.decay_every == opt.decay_every-1:   
                del loss
                scores,prec_,recall_ ,_ss= val(model,dataset)
                vis.log({' epoch:':epoch,' lr: ':lr,'scores':scores,'prec':prec_,'recall':recall_,'ss':_ss,'scores_train':score_meter.value()[0],'loss':loss_meter.value()[0]})

                if scores>best_score:
                    best_score = scores
                    best_path = model.save(name = str(scores),new=True)
               
                if scores < best_score:
                    model.load(best_path,change_opt=False)
                    lr = lr * opt.lr_decay
                    if lr2==0:lr2=1e-4
                    else : lr2 = lr2*0.5
                    optimizer = model.get_optimizer(lr,lr2,0)
                             
                pre_loss = loss_meter.value()[0]
                loss_meter.reset()
                score_meter.reset()
                

if __name__=="__main__":
    fire.Fire()  
