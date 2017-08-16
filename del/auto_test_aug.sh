#python test_aug.py main --model='MultiCNNTextBNDeep'  --batch-size=512  --model-path='checkpoints/MultiCNNTextBNDeep_char_0.387384946476'  --val=True --zhuge=True --type_='char'
#python test_aug.py main --model='CNNText_inception'  --batch-size=512  --model-path='checkpoints/CNNText_tmp_word_0.41254624328'  --val=True --zhuge=True
#python test_aug.py main --model='RCNN'  --batch-size=512  --model-path='checkpoints/RCNN_word_0.413446202556'  --val=True --zhuge=True
#python test_aug.py main --model='LSTMText'  --batch-size=512  --model-path='checkpoints/LSTMText_word_0.412004681315'  --val=True --zhuge=True
#python test_aug.py main --model='RCNN'  --batch-size=512  --model-path='checkpoints/RCNN_char_0.398541405915'  --val=True --zhuge=True --type_='char'
#python test_aug.py main --model='LSTMText'  --batch-size=512  --model-path='checkpoints/LSTMText_word_0.413681107036'  --val=True --zhuge=True

python test_aug_multimodel.py main --model='MultiModelAll'  --batch-size=512  --model-path='checkpoints/MultiModelAll_word_0.417185977233'  --val=True --zhuge=True --loss='bceloss'