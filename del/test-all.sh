# LSTM
#python2 test.1.py main --model='LSTMText'  --batch-size=512  --model-path='checkpoints/LSTMText_word_0.411994005382' --result-path='/data_ssd/zhihu/result/LSTMText0.4119_word_val.pth'  --val=True --zhuge=True

python2 test.1.py main --model='LSTMText'  --batch-size=256 --type_=char --model-path='checkpoints/LSTMText_char_0.403192339135' --result-path='/data_ssd/zhihu/result/LSTMText0.4031_char_val.pth'  --val=True --zhuge=True

# Inception
python2 test.1.py main --model='CNNText_inception'  --batch-size=512  --model-path='checkpoints/CNNText_tmp_word_0.41096749885' --result-path='/data_ssd/zhihu/result/CNNText_tmp0.4109_word_val.pth'  --val=True --zhuge=True

python2 test.1.py main --model='CNNText_inception' --type_=char  --batch-size=512  --model-path='checkpoints/CNNText_tmp_char_0.402429167301' --result-path='/data_ssd/zhihu/result/CNNText_tmp0.4024_char_val.pth'  --val=True --zhuge=True

#RCNN
python2 test.1.py main --model='RCNN'  --batch-size=512  --model-path='checkpoints/RCNN_word_0.411511574999' --result-path='/data_ssd/zhihu/result/RCNN_0.4115_word_val.pth'  --val=True --zhuge=True

python2 test.1.py main --model='RCNN'  --batch-size=512  --model-path='checkpoints/RCNN_char_0.403710422571' --result-path='/data_ssd/zhihu/result/RCNN_0.4037_char_val.pth'  --val=True --zhuge=True

# DeepText

python2 test.1.py main --model='MultiCNNTextBNDeep'  --batch-size=512  --model-path='checkpoints/MultiCNNTextBNDeep_word_0.410330780091' --result-path='/data_ssd/zhihu/result/DeepText0.4103_word_val.pth'  --val=True --zhuge=True

# MultiModel
python2 test.3.py main --model='MultiModelAll'  --batch-size=64  --model-path='checkpoints/MultiModelAll_word_0.419866393964' --result-path='/data_ssd/zhihu/result/MultiModel_0.41987_word_val.pth'  --val=True --zhuge=True --loss='bceloss'

##########################################TEST#################
# LSTM
python2 test.1.py main --model='LSTMText'  --batch-size=512  --model-path='checkpoints/LSTMText_word_0.411994005382' --result-path='/data_ssd/zhihu/result/LSTMText0.4119_word_test.pth'  --val=False --zhuge=True

python2 test.1.py main --model='LSTMText'  --batch-size=256 --type_=char --model-path='checkpoints/LSTMText_char_0.403192339135' --result-path='/data_ssd/zhihu/result/LSTMText0.4031_char_test.pth'  --val=False --zhuge=True

# Inception
python2 test.1.py main --model='CNNText_inception'  --batch-size=512  --model-path='checkpoints/CNNText_tmp_word_0.41096749885' --result-path='/data_ssd/zhihu/result/CNNText_tmp0.4109_word_test.pth'  --val=False --zhuge=True

python2 test.1.py main --model='CNNText_inception' --type_=char  --batch-size=512  --model-path='checkpoints/CNNText_tmp_char_0.402429167301' --result-path='/data_ssd/zhihu/result/CNNText_tmp0.4024_char_test.pth'  --val=False --zhuge=True

#RCNN
python2 test.1.py main --model='RCNN'  --batch-size=512  --model-path='checkpoints/RCNN_word_0.411511574999' --result-path='/data_ssd/zhihu/result/RCNN_0.4115_word_test.pth'  --val=False --zhuge=True

python2 test.1.py main --model='RCNN'  --batch-size=512  --model-path='checkpoints/RCNN_char_0.403710422571' --result-path='/data_ssd/zhihu/result/RCNN_0.4037_char_test.pth'  --val=False --zhuge=True

# DeepText

python2 test.1.py main --model='MultiCNNTextBNDeep'  --batch-size=512  --model-path='checkpoints/MultiCNNTextBNDeep_word_0.410330780091' --result-path='/data_ssd/zhihu/result/DeepText0.4103_word_test.pth'  --val=False --zhuge=True

#MultiModelALl

python2 test.3.py main --model='MultiModelAll'  --batch-size=64  --model-path='checkpoints/MultiModelAll_word_0.419866393964' --result-path='/data_ssd/zhihu/result/MultiModel_0.41987_word_test.pth'  --val=False --zhuge=True --loss='bceloss'

