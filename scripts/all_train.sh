# LSTM
 python main.py main --max_epoch=2 --plot_every=100 --env='MultiCNNText' --weight=1 --model='MultiCNNTextBNDeep'  --batch-size=64  --lr=0.001 --lr2=0.0005 --lr_decay=0.8 --decay_every=10000  --title-dim=250 --content-dim=250    --weight-decay=0 --type_='word' --debug-file='/tmp/debug'  --linear-hidden-size=2000 --zhuge=True  

python main.py main --max_epoch=2 --plot_every=100 --env='lstm_word' --weight=1 --model='LSTMText'  --batch-size=64  --lr=0.001 --lr2=0.0000 --lr_decay=0.8 --decay_every=5400  --type_='word'   --zhuge=True --linear-hidden-size=1000 --hidden-size=256 --kmax-pooling=1

python main.py main --max_epoch=2 --plot_every=100 --env='lstm_char' --weight=1 --model='LSTMText'  --batch-size=64  --lr=0.001 --lr2=0 --lr_decay=0.8 --decay_every=5400  --type_='char'   --zhuge=True --linear-hidden-size=1000 --hidden-size=256 --kmax-pooling=1  


# RCNN
python main.py main --max_epoch=2 --plot_every=100 --env='rcnn_char' --weight=1 --model='RCNN'  --batch-size=64  --lr=0.001 --lr2=0  --lr_decay=0.5 --decay_every=5400  --title-dim=500 --content-dim=500  --type_='char' --zhuge=True --kernel-size=3 --kmax-pooling=1 --linear-hidden-size=2000 --debug-file='/tmp/debugrcnn' --hidden-size=256 --num-layers=2 --debug=True


python main.py main --max_epoch=2 --plot_every=100 --weight=1 --model='RCNN'  --env='rcnn-word' --batch-size=64  --lr=0.001 --lr2=0.000 --lr_decay=0.8 --decay_every=5000  --title-dim=500 --content-dim=500  --type_='word'  --kernel-size=2 --kmax-pooling=1 --linear-hidden-size=2000 --debug-file='/tmp/debugrcnn' --hidden-size=256 --decay_every=5000 --num-workers=4 --zhuge=True

python main.py main --max_epoch=6 --plot_every=100 --env='multimodel_raw' --weight=1 --model='MultiModel'  --zhuge=True --num-workers=4 --batch-size=32 --model-path=None --lr2=0 --lr=1e-3 --lr-decay=0.5 --loss='bceloss'  --decay-every=5400




















%run test.1.py main --model='LSTMText'  --batch-size=256 --type_=char --model-path='checkpoints/LSTMText_char_0.403192339135' --result-path='checkpoints/result/LSTMText0.4031_char_val.pth'  --val=True --zhuge=True

# Inception
%run test.1.py main --model='CNNText_inception'  --batch-size=512  --model-path='checkpoints/CNNText_tmp_word_0.41096749885' --result-path='checkpoints/result/CNNText_tmp0.4109_word_val.pth'  --val=True --zhuge=True

%run test.1.py main --model='CNNText_inception' --type_=char  --batch-size=512  --model-path='checkpoints/CNNText_tmp_char_0.402429167301' --result-path='checkpoints/result/CNNText_tmp0.4024_char_val.pth'  --val=True --zhuge=True

#RCNN
%run test.1.py main --model='RCNN'  --batch-size=512  --model-path='checkpoints/RCNN_word_0.411511574999' --result-path='checkpoints/result/RCNN_0.4115_word_val.pth'  --val=True --zhuge=True

%run test.1.py main --model='RCNN'  --batch-size=512  --model-path='checkpoints/RCNN_char_0.403710422571' --result-path='checkpoints/result/RCNN_0.4037_char_val.pth'  --val=True --zhuge=True

# DeepText

%run test.1.py main --model='MultiCNNTextBNDeep'  --batch-size=512  --model-path='checkpoints/MultiCNNTextBNDeep_word_0.410330780091' --result-path='checkpoints/result/DeepText0.4103_word_val.pth'  --val=True --zhuge=True

%run test.1.py main --model='CNNText_inception'  --batch-size=512  --model-path='checkpoints/CNNText_tmp_word_0.41096749885' --result-path='checkpoints/result/C NNText_tmp0.4109_word_val.pth'  --val=True --zhuge=True

%run test.1.py main --model='CNNText_inception'  --batch-size=512  --model-path='checkpoints/CNNText_tmp_word_0.41096749885' --result-path='checkpoints/result/C NNText_tmp0.4109_word_val.pth'  --val=True --zhuge=True

%run test.1.py main --model='CNNText_inception'  --batch-size=512  --model-path='checkpoints/CNNText_tmp_word_0.41096749885' --result-path='checkpoints/result/C NNText_tmp0.4109_word_val.pth'  --val=True --zhuge=True