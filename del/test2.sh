
# ## deeptext

# %run -i test.1.py main --model="MultiCNNTextBNDeep" --model_path="checkpoints/MultiCNNTextBNDeep_word_0.40940922954" --result_path="/data_ssd/zhihu/result/tmp/MultiCNNTextBNDeep_word_weight5_0.409409_val.pth"  --val=True --batch_size=512


# %run -i test.1.py main --model="MultiCNNTextBNDeep" --model_path="checkpoints/MultiCNNTextBNDeep_word_0.40940922954" --result_path="/data_ssd/zhihu/result/tmp/MultiCNNTex
#    ...: tBNDeep_word_weight5_0.409409_val.pth"  --val=True --batch_size=512


## rcnn

python2 test.1.py main --model="RCNN" --model_path="checkpoints/RCNN_word_0.412260353642" --result_path="/data_ssd/zhihu/result/tmp/RCNN_0.41226_weight5_val.pth"  --val=True --batch_size=512



python2 test.1.py main --model="RCNN" --model_path="checkpoints/RCNN_word_0.412260353642" --result_path="/data_ssd/zhihu/result/tmp/RCNN_0.41226_weight5_test.pth"  --val=False --batch_size=512


## lstm

python2 test.1.py main --model="LSTMText" --model_path="checkpoints/LSTMText_word_0.4123032919" --result_path="/data_ssd/zhihu/result/tmp/LSTMText_0.41230_weight5_val.pth"  --val=True --batch_size=512

python2 test.1.py main --model="LSTMText" --model_path="checkpoints/LSTMText_word_0.4123032919" --result_path="/data_ssd/zhihu/result/tmp/LSTMText_0.41230_weight5_test.pth"  --val=False --batch_size=512
## inception


python2 test.1.py main --model="CNNText_inception" --model_path="checkpoints/CNNText_inception_word_0.409912169383" --result_path="/data_ssd/zhihu/result/tmp/inception_0.409912_weight5_val.pth"  --val=True --batch_size=512

python2 test.1.py main --model="CNNText_inception" --model_path="checkpoints/CNNText_inception_word_0.409912169383" --result_path="/data_ssd/zhihu/result/tmp/inception_0.409912_weight5_test.pth"  --val=False --batch_size=512