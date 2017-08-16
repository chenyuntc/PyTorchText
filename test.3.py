#encoding:utf-8
# from torch.utils import data

'''
专门为multimodel而写 生成pth
'''
import torch as t
import numpy as np
from config import opt
import models
import json
import fire
import csv
import tqdm
from torch.autograd import Variable
def load_data(type_='char'):
    with open(opt.labels_path) as f:
        labels_ = json.load(f)
    question_d = np.load(opt.test_data_path)
    # if type_ == 'char':
    #     test_data_title,test_data_content =\
    #          question_d['title_char'],question_d['content_char']

    # elif type_ == 'word':
    #     test_data_title,test_data_content =\
    #          question_d['title_word'],question_d['content_word']

    index2qid = question_d['index2qid'].item()
    return (question_d['title_char'],question_d['content_char']),( question_d['title_word'],question_d['content_word']),index2qid,labels_['id2label']
def write_csv(result,index2qid,labels):
    f=open(opt.result_path, "wa")
    csv_writer = csv.writer(f, dialect="excel")
    rows=[0 for _ in range(result.shape[0])]
    for i in range(result.shape[0]):
        row=[index2qid[i]]+[labels[str(int(i_))] for i_ in result[i]]
        rows[i]=row
    csv_writer.writerows(rows)
def dotest(model,title,content):
    title,content = (Variable(t.from_numpy(title[0]).long().cuda(),volatile=True),Variable(t.from_numpy(title[1]).long().cuda(),volatile=True)),(Variable(t.from_numpy(content[0]).long().cuda(),volatile=True),Variable(t.from_numpy(content[1]).long().cuda(),volatile=True))
    score = model(title,content)
    probs=t.nn.functional.sigmoid(score)
    return probs.data.cpu().numpy()
    
def main(**kwargs):
    opt.parse(kwargs)
    # opt.model_names=['MultiCNNTextBNDeep','LSTMText','CNNText_inception','RCNN']
    # opt.model_paths = ['checkpoints/MultiCNNTextBNDeep_0.37125473788','checkpoints/LSTMText_word_0.381833388089','checkpoints/CNNText_tmp_0.376364647145','checkpoints/RCNN_char_0.3456599248']
    # opt.model_names=['MultiCNNTextBNDeep','CNNText_inception',
    # #'RCNN',
    # 'LSTMText','CNNText_inception']
    # opt.model_paths = ['checkpoints/MultiCNNTextBNDeep_word_0.410330780091','checkpoints/CNNText_tmp_word_0.41096749885',
    # #'checkpoints/RCNN_word_0.411511574999',
    # 'checkpoints/LSTMText_word_0.411994005382','checkpoints/CNNText_tmp_char_0.402429167301'] 


    # opt.model_names=['MultiCNNTextBNDeep','RCNN','LSTMText','RCNN','CNNText_inception']
    # opt.model_paths = ['checkpoints/MultiCNNTextBNDeep_word_0.41124002492','checkpoints/RCNN_word_0.411511574999','checkpoints/LSTMText_word_0.411994005382','checkpoints/RCNN_char_0.403710422571','checkpoints/CNNText_tmp_char_0.402429167301']
    # opt.model_path='checkpoints/MultiModelAll2_word_0.425600838271'
    # opt.model_names=['MultiCNNTextBNDeep',
    # 'LSTMText',
    # 'CNNText_inception',
    # 'RCNN',
    # ]
    # opt.model_paths = ['checkpoints/MultiCNNTextBNDeep_word_0.410330780091',
    # # 'checkpoints/CNNText_tmp_word_0.41096749885',
    # #'checkpoints/RCNN_word_0.411511574999',
    # 'checkpoints/LSTMText_word_0.411994005382',
    # 'checkpoints/CNNText_tmp_char_0.402429167301',
    # 'checkpoints/RCNN_char_0.403710422571'
    # ]
    # opt.model_path='checkpoints/MultiModelAll_word_0.421331405593'
    #############################################################################################
    # opt.model_names=['MultiCNNTextBNDeep','RCNN',
    # #'RCNN',
    # 'LSTMText','RCNN']
    # opt.model_paths = ['checkpoints/MultiCNNTextBNDeep_word_0.410011182415','checkpoints/RCNN_word_0.413446202556',
    # 'checkpoints/LSTMText_word_0.413681107036',
    # #'checkpoints/RCNN_word_0.411511574999',
    # 'checkpoints/RCNN_char_0.398378946148'] 

    # opt.model_path='checkpoints/MultiModelAll_word_0.423535867989'
#########################################################################################################################


    # opt.model_names=['MultiCNNTextBNDeep','FastText3','LSTMText']
    # opt.model_paths = ['checkpoints/MultiCNNTextBNDeep_word_0.410011182415','checkpoints/FastText3_word_0.40810787337',
    # 'checkpoints/LSTMText_word_0.413681107036']
    # opt.model_path='checkpoints/MultiModelAll_word_0.416523282174'




################################################################################3

    

 #   opt.model_names=['MultiCNNTextBNDeep','FastText3','LSTMText','CNNText_inception']
  #  opt.model_paths = ['checkpoints/MultiCNNTextBNDeep_word_0.41124002492','checkpoints/FastText3_word_0.40810787337','checkpoints/LSTMText_word_0.413681107036','checkpoints/CNNText_tmp_char_0.402429167301']
  #  opt.model_path='checkpoints/MultiModelAll2_word_0.419088407885'
#####################################################################

    opt.model_names=['CNNText_inception','FastText3','RCNN']
    opt.model_paths = ['checkpoints/CNNText_tmp_word_0.41254624328','checkpoints/FastText3_word_0.409160833419',
    'checkpoints/RCNN_word_0.413446202556']
    opt.model_path='checkpoints/MultiModelAll_word_0.419245894992'
    
    ################################################################
    model = getattr(models,opt.model)(opt).cuda().eval()
    if opt.model_path is not None:
        model.load(opt.model_path)
    model=model.eval()
    opt.parse(kwargs)
    
    test_data_title,test_data_content,index2qid,labels=load_data(type_=opt.type_)
    Num=len(test_data_title[0])
    result=np.zeros((Num,1999))
    for i in tqdm.tqdm(range(Num)):
        if i%opt.batch_size==0 and i>0:
            title=np.array(test_data_title[0][i-opt.batch_size:i]),np.array(test_data_title[1][i-opt.batch_size:i])
            content=np.array(test_data_content[0][i-opt.batch_size:i]),np.array(test_data_content[1][i-opt.batch_size:i])
            result[i-opt.batch_size:i,:]=dotest(model,title,content)  
    if Num%opt.batch_size!=0:
        title=np.array(test_data_title[0][opt.batch_size*(Num/opt.batch_size):]),np.array(test_data_title[1][opt.batch_size*(Num/opt.batch_size):])
        content=np.array(test_data_content[0][opt.batch_size*(Num/opt.batch_size):]) ,np.array(test_data_content[1][opt.batch_size*(Num/opt.batch_size):]) 
        result[opt.batch_size*(Num/opt.batch_size):,:]=dotest(model,title,content) 
    t.save(t.from_numpy(result).float(),opt.result_path)
    
if __name__=='__main__':
    fire.Fire()
    
    