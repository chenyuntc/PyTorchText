#coding:utf8

import json
label_path =   '/mnt/7/zhihu/ieee_zhihu_cup/data/labels.json'
test_data_path='/mnt/7/zhihu/ieee_zhihu_cup/data/test.npz' 
def ensamble(file1,file2,label_path=label_path,     test_data_path=test_data_path,result_csv=None):
    import torch as t 
    import numpy as np
    if result_csv is None:
        import time
        result_csv = time.strftime('%y%m%d_%H%M%S.csv')
    a = t.load(file1)
    b = t.load(file2)
    r = 9.0*a+b
    result = r.topk(5,1)[1]
    
    index2qid = np.load(test_data_path)['index2qid'].item()
    with open(label_path) as f:   label2qid = json.load(f)['id2label']
    rows = range(result.size(0))
    for ii,item in enumerate(result):
        rows[ii] = [index2qid[ii]] + [label2qid[str(_)] for _ in item ]
    import csv
    with open(result_csv,'w') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    

if __name__ == '__main__':
    import fire
    fire.Fire()

files = ['CNNText_tmp0.4024_char_test.pth',
 'CNNText_tmp0.4024_char_val.pth',
 'DeepText0.4103_word_test.pth',
 'Inception0.4110_word.pth',
 'LSTMText0.4119_word.pth',
 'LSTMText0.4031_char_test.pth',
 'LSTMText0.4119_word_test.pth',
 'DeepText0.4103_word_val.pth',
 'CNNText_tmp0.4109_word_val.pth',
 'LSTMText0.4119_word_val.pth',
 'RCNN_0.4115_word_test.pth',
 'RCNN_0.4037_char_val.pth',
 'LSTMText0.4031_char_val.pth',
 'RCNN_0.4115_word_val.pth',
 'RCNN_0.4037_char_test.pth',
 'CNNText_tmp0.4109_word_test.pth']
