

# 中文用户请查看[readme-zh.md](readme-zh.md)

This is the solution for [Zhihu Machine Learning Challenge 2017](https://biendata.com/competition/zhihu/). We won the champion out of 963 teams.

## 1. Setup
- install PyTorch from [pytorch.org](http://pytorch.org/) (Python 2, CUDA)
- install other depencies:
    ```sh
    pip2 install -r requirements.txt
    ```
You may need `tf.contrib.keras.preprocessing.sequence.pad_sequences` for data preprocessing.
- start visdom for visualization:
    ```sh
    python2 -m visdom.server
    ```


## 2. Data Preprocessing

__Modify the data path in the related file__

###  2.1 wordvector file -> numpy file
```sh
python scripts/data_process/embedding2matrix.py main char_embedding.txt char_embedding.npz 
python scripts/data_process/embedding2matrix.py main word_embedding.txt word_embedding.npz 
```

### 2.2  question set -> numpy file

it's memory consuming , make sure you have memory larger than 32G.
```sh
python scripts/data_process/question2array.py main question_train_set.txt train.npz
python scripts/data_process/question2array.py main question_eval_set.txt test.npz
```
### 2.3 label -> json
```sh
python scripts/data_process/label2id.py main question_topic_train_set.txt labels.json
```
### 2.4 validation data


```sh
python scripts/data_process/get_val.py 
``` 

## 3. Training

modify `config.py` for model path

Path to the models we used:
- CNN:`models/MultiCNNTextBNDeep.py`
- RNN（LSTM）:`models/LSTMText.py`
- RCNN: `models/RCNN.py`
- inception: `models/CNNText_inception.py`
- FastText: `models/FastText3.py`


### 3.1 Trian model without data augumentation
```sh
# LSTM char
python2 main.py main --max_epoch=5 --plot_every=100 --env='lstm_char' --weight=1 --model='LSTMText'  --batch-size=128  --lr=0.001 --lr2=0 --lr_decay=0.5 --decay_every=10000  --type_='char'   --zhuge=True --linear-hidden-size=2000 --hidden-size=256 --kmax-pooling=3   --num-layers=3  --augument=False

# LSTM word
python2 main.py main --max_epoch=5 --plot_every=100 --env='lstm_word' --weight=1 --model='LSTMText'  --batch-size=128  --lr=0.001 --lr2=0.0000 --lr_decay=0.5 --decay_every=10000  --type_='word'   --zhuge=True --linear-hidden-size=2000 --hidden-size=320 --kmax-pooling=2  --augument=False

#  RCNN char
python2 main.py main --max_epoch=5 --plot_every=100 --env='rcnn_char' --weight=1 --model='RCNN'  --batch-size=128  --lr=0.001 --lr2=0 --lr_decay=0.5 --decay_every=5000  --title-dim=1024 --content-dim=1024  --type_='char' --zhuge=True --kernel-size=3 --kmax-pooling=2 --linear-hidden-size=2000 --debug-file='/tmp/debugrcnn' --hidden-size=256 --num-layers=3 --augument=False

# RCNN word
main.py main --max_epoch=5 --plot_every=100 --env='RCNN-word' --weight=1 --model='RCNN'  --zhuge=True --num-workers=4 --batch-size=128 --model-path=None --lr2=0 --lr=1e-3 --lr-decay=0.8  --decay-every=5000  --title-dim=1024 --content-dim=512  --kernel-size=3 --debug-file='/tmp/debugrc'  --kmax-pooling=1 --type_='word' --augument=False
# CNN word
 python main.py main --max_epoch=5 --plot_every=100 --env='MultiCNNText' --weight=1 --model='MultiCNNTextBNDeep'  --batch-size=64  --lr=0.001 --lr2=0.000 --lr_decay=0.8 --decay_every=10000  --title-dim=250 --content-dim=250    --weight-decay=0 --type_='word' --debug-file='/tmp/debug'  --linear-hidden-size=2000 --zhuge=True  --augument=False

# inception word
python2 main.py main --max_epoch=5 --plot_every=100 --env='inception-word' --weight=1 --model='CNNText_inception'  --zhuge=True --num-workers=4 --batch-size=512 --model-path=None --lr2=0 --lr=1e-3 --lr-decay=0.8  --decay-every=2500 --title-dim=1200 --content-dim=1200 --type_='word' --augument=False                                                   
# inception char
python2 main.py main --max_epoch=5 --plot_every=100 --env='inception-char' --weight=1 --model='CNNText_inception'  --zhuge=True --num-workers=4 --batch-size=512 --model-path=None --lr2=0 --lr=1e-3 --lr-decay=0.8  --decay-every=2500 --title-dim=1200 --content-dim=1200 --type_='char'   --augument=False

# FastText3 word
python2 main.py main --max_epoch=5 --plot_every=100 --env='fasttext3-word' --weight=5 --model='FastText3' --zhuge=True --num-workers=4 --batch-size=512  --lr2=1e-4 --lr=1e-3 --lr-decay=0.8  --decay-every=2500 --linear_hidden_size=2000 --type_='word'  --debug-file=/tmp/debugf --augument=False                           

```

In most cases, the score could be boosted by finetune. for example:

```sh
python2 main.py main --max_epoch=2 --plot_every=100 --env='LSTMText-word-ft' --model='LSTMText'  --zhuge=True --num-workers=4 --batch-size=256 --model-path=None --lr2=5e-5 --lr=5e-5 --decay-every=5000 --type_='word'  --model-path='checkpoints/LSTMText_word_0.409196378421'                       
```

### 3.2 train models with data augumentation

Add `--augument` in the training command.



### 3.3 scores

|model|score|
:---:|:----:
CNN_word|0.4103
RNN_word|0.4119
RCNN_word|0.4115
Inceptin_word|0.4109
FastText_word|0.4091
RNN_char|0.4031
RCNN_char|0.4037
Inception_char|0.4024
RCNN_word_aug|0.41344
CNN_word_aug|0.41051
RNN_word_aug|0.41368
Incetpion_word_aug|0.41254
FastText3_word_aug|0.40853
CNN_char_aug|0.38738
RCNN_char_aug|0.39854

with model ensemble, it can get up to 0.433.


## 4 Test and Submit
### 4.1 Test


- model: include `LSTMText`,`RCNN`,`MultiCNNTextBNDeep`,`FastText3`,`CNNText_inception`
- model-path: path to the pretrained model
- result-path: where to save the model
- val: test the val set or the test set..

```sh
# LSTM
python2 test.1.py main --model='LSTMText'  --batch-size=512  --model-path='checkpoints/LSTMText_word_0.411994005382' --result-path='/data_ssd/zhihu/result/LSTMText0.4119_word_test.pth'  --val=False --zhuge=True

python2 test.1.py main --model='LSTMText'  --batch-size=256 --type_=char --model-path='checkpoints/LSTMText_char_0.403192339135' --result-path='/data_ssd/zhihu/result/LSTMText0.4031_char_test.pth'  --val=False --zhuge=True
 
#RCNN
python2 test.1.py main --model='RCNN'  --batch-size=512  --model-path='checkpoints/RCNN_word_0.411511574999' --result-path='/data_ssd/zhihu/result/RCNN_0.4115_word_test.pth'  --val=False --zhuge=True

python2 test.1.py main --model='RCNN'  --batch-size=512  --model-path='checkpoints/RCNN_char_0.403710422571' --result-path='/data_ssd/zhihu/result/RCNN_0.4037_char_test.pth'  --val=False --zhuge=True

# DeepText

python2 test.1.py main --model='MultiCNNTextBNDeep'  --batch-size=512  --model-path='checkpoints/MultiCNNTextBNDeep_word_0.410330780091' --result-path='/data_ssd/zhihu/result/DeepText0.4103_word_test.pth'  --val=False --zhuge=True
# more to go ...
```


### 4.2 ensemble
See `notebooks/val_ensemble.ipynb` and `notebooks/test_ensemble.ipynb` for more detail 



### 5 Main files
- `main.py`: main(for training)
- `config.py`: config file
- `test.1.py`: for test
- `data/`: for data loader
- `scripts/`: for data preprocessing
- `utils/` : including calculate score and wrapper for visualization.
- `models/`: models
    - `models/BasicModel`: Base model for models.
    - `models/MultiCNNTextBNDeep`: CNN
    - `models/LSTMText`: RNN
    - `models/RCNN`: RCNN
    - `models/CNNText_inception` Inception
    - `models/MultiModelALL` 和`models/MultiModelAll2`
    -  other model     
- `rep.py`: code for reproducing.
- `del/`: methods fail or not used.
- `notebooks/`: notebooks.

### Pretrained model
https://pan.baidu.com/s/1mjVtJGs passwd: tayb
