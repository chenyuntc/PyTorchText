#encoding:utf-8
from torch.utils import data
import torch as t
import numpy as np
from config import opt
import models
import json
import fire
import csv
from torch.autograd import Variable
import tqdm
from glob import glob
def load_data(test,labels_file):
    '''
    data_root="/data/text/zhihu/result/*_test.pth"
    labels_file="/home/a/code/pytorch/zhihu/ddd/labels.json"
    test="/home/a/code/pytorch/zhihu/ddd/test.npz"
    '''
    index2qid=np.load(test)['index2qid'].item()
    with open(labels_file) as f:
        labels= json.load(f)['id2label']
    #test_data_num=217360
    #model_num=len(result_files_path)
    #test_data=t.zeros(test_data_num,1999*model_num)
    #for i in range(model_num):
    #    test_data[:,i*1999:i*1999+1999]=t.load(result_files_path[i]).float()
    return index2qid,labels

def test_hyper():
    data_root="/data/text/zhihu/result/tmp/"
    labels_file="/home/a/code/pytorch/zhihu/ddd/labels.json"
    test="/home/a/code/pytorch/zhihu/ddd/test.npz"
    index2qid,labels=load_data(test,labels_file)
    result_files_path=glob(data_root+"*val.pth")
    result_files_path.sort()
    result=0
    #a=[0.19447573541703533, 0.4302956826328484, 0.24415100069297443, 3.1941015222067826, 1.2995016739712848, 0.8448979155201735, 1.54270375208129]
    # 0.4292096367086292
    #a=[1.7212561973117408,1.7058806674841556,1.3764529360325304,1.1685564476646357,1.7403842367575626,0.7891955839006122, 1.6554675840993296]
    # 大约0.4271
    a=[0.22478083908058233,0.6963877252668917,0.056146692013221464,2.76910389573104318311,2.842203642680702,1.1596530417845,3.0584950428073454]
    # 大约0.4293322872472573 
    #a=[1,1,1,1,1,1,1]
    for i,a_ in enumerate (a):
        print "load ",result_files_path[i],a_
        result=result+a_*t.load(result_files_path[i].replace("val","test")).float()
    write_csv(result,index2qid,labels)
        
        
def main(**kwargs):
    opt.parse(kwargs,print_=False)
    model = getattr(models,opt.model)(opt).cuda().eval()
    if opt.model_path is not None:
        print "loding model",opt.model_path
        model.load(opt.model_path)
    #model=model.eval()
    opt.parse(kwargs,print_=False)
    data_root="/data/text/zhihu/result/"
    labels_file="/home/a/code/pytorch/zhihu/ddd/labels.json"
    test="/home/a/code/pytorch/zhihu/ddd/test.npz"
    index2qid,labels=load_data(test,labels_file)
    result_files_path=glob(data_root+"*val.pth")
    weights=t.nn.functional.softmax(model.weights)
    print "max of weights",weights.max()
    print "min of weights",weights.min()
    print "mean of weights",weights.mean()
    Num=217360
    tmp_result=t.zeros((Num,1999))
    print result_files_path
    for i in range(len(result_files_path)):
        tmpdata=t.load(result_files_path[i].replace("val","test")).float()
        for j in tqdm.tqdm(range(Num)):
            if j%opt.batch_size==0 and j>0:
                data=tmpdata[j-opt.batch_size:j] 
                weights_=weights[:,i].contiguous().view(1,1999).expand_as(data).data.cpu()
                tmp_result[j-opt.batch_size:j,:]=tmp_result[j-opt.batch_size:j,:]+weights_*data 
            if j%Num-1 ==0:
                print j,"have done"
        data=tmpdata[j-opt.batch_size:j] 
        weights_=weights[:,i].contiguous().view(1,1999).expand_as(data).data.cpu()
        tmp_result[j-opt.batch_size:j,:]=tmp_result[j-opt.batch_size:j,:]+weights_*data
    write_csv(tmp_result,index2qid,labels)
def write_csv(result,index2qid,labels):
    #result_=result.sort(dim=1,descending=True)
    #del result
    f=open(opt.result_path, "wa")
    csv_writer = csv.writer(f, dialect="excel")
    rows=[0 for _ in range(result.size(0))]
    #result_=result.numpy()
    for i in range(result.size(0)):
        tmp=result[i].sort(dim=0,descending=True)
        tmp=tmp[1][:5]
        row=[index2qid[i]]+[labels[str(int(i_))] for i_ in tmp]
        rows[i]=row
    csv_writer.writerows(rows)    
def dotest(weights,data,i):
    return weights[:,i].contiguous().view(1,1999).expand_as(data).data.cpu()*data
    
    
    
if __name__=='__main__':
    fire.Fire()


    