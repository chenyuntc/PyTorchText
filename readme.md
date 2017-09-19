

[TOC]

这是首届知乎[看山杯](https://biendata.com/competition/zhihu/)冠军init的解决方案，关于参赛方法，请参阅[知乎专栏的文章](https://zhuanlan.zhihu.com/p/28923961)

## 1. 环境配置
本程序基于PyTorch，需要从[官网](http://pytorch.org/)下载指定版本的PyTorch（2.7，CUDA）. 我用的版本[`0.1.12.2`](http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl),现在`0.2`已经出来了,不知道是否会有兼容性的问题.
另外还需要安装以下依赖：
```sh
pip2 install -r requirements.txt
```
主要包括以下工具:

- PyTorch工具：[PyTorchNet](https://github.com/pytorch/tnt)
- 进度条工具：[tqdm](https://github.com/tqdm/tqdm)
- 命令行工具：[fire](github.com/google/python-fire)
- 可视化工具: [visdom](https://github.com/facebookresearch/visdom)
- 交互式调试工具: [ipdb](https://github.com/gotcha/ipdb)

数据预处理中还用到了`tf.contrib.keras.preprocessing.sequence.pad_sequences`,需要安装TensorFlow,当然`numpy`是必不可少的.


安装完上述依赖之后，启动可视化工具visdom 服务
```sh
python2 -m visdom.server
```


## 2. 数据预处理

__注意修改文件路径__

###  2.1 词向量转成numpy数组
```sh
python scripts/data_process/embedding2matrix.py main char_embedding.txt char_embedding.npz 
python scripts/data_process/embedding2matrix.py main word_embedding.txt word_embedding.npz 
```

### 2.2  问题转成numpy 数组

这一步很耗内存，请确保内存>32G
```sh
python scripts/data_process/question2array.py main question_train_set.txt train.npz
python scripts/data_process/question2array.py main question_eval_set.txt test.npz
```
### 2.3 处理label，转成json
```sh
python scripts/data_process/label2id.py main question_topic_train_set.txt labels.json
```
### 2.4 生成验证集

从训练集中抽取一部分的数据生成验证集, 这部分代码是从ipython中备份的,__注意修改代码中的数据存放路径__ .

```sh
python scripts/data_process/get_val.py 
``` 

## 3. 训练模型

注意修改`config.py`中文件的路径

主要用到了五个模型
- CNN:`models/MultiCNNTextBNDeep.py`
- RNN（LSTM）:`models/LSTMText.py`
- RCNN: `models/RCNN.py`
- inception: `models/CNNText_inception.py`
- FastText: `models/FastText3.py`

分别训练两个对应的word模型和char模型

### 3.1 训练不进行数据增强的模型
```sh
# 训练LSTM char
python2 main.py main --max_epoch=5 --plot_every=100 --env='lstm_char' --weight=1 --model='LSTMText'  --batch-size=128  --lr=0.001 --lr2=0 --lr_decay=0.5 --decay_every=10000  --type_='char'   --zhuge=True --linear-hidden-size=2000 --hidden-size=256 --kmax-pooling=3   --num-layers=3  --augument=False

# 训练LSTM word
python2 main.py main --max_epoch=5 --plot_every=100 --env='lstm_word' --weight=1 --model='LSTMText'  --batch-size=128  --lr=0.001 --lr2=0.0000 --lr_decay=0.5 --decay_every=10000  --type_='word'   --zhuge=True --linear-hidden-size=2000 --hidden-size=320 --kmax-pooling=2  --augument=False

# 训练 RCNN char
python2 main.py main --max_epoch=5 --plot_every=100 --env='rcnn_char' --weight=1 --model='RCNN'  --batch-size=128  --lr=0.001 --lr2=0 --lr_decay=0.5 --decay_every=5000  --title-dim=1024 --content-dim=1024  --type_='char' --zhuge=True --kernel-size=3 --kmax-pooling=2 --linear-hidden-size=2000 --debug-file='/tmp/debugrcnn' --hidden-size=256 --num-layers=3 --augument=False

# 训练RCNN word
main.py main --max_epoch=5 --plot_every=100 --env='RCNN-word' --weight=1 --model='RCNN'  --zhuge=True --num-workers=4 --batch-size=128 --model-path=None --lr2=0 --lr=1e-3 --lr-decay=0.8  --decay-every=5000  --title-dim=1024 --content-dim=512  --kernel-size=3 --debug-file='/tmp/debugrc'  --kmax-pooling=1 --type_='word' --augument=False
# 训练CNN word
 python main.py main --max_epoch=5 --plot_every=100 --env='MultiCNNText' --weight=1 --model='MultiCNNTextBNDeep'  --batch-size=64  --lr=0.001 --lr2=0.000 --lr_decay=0.8 --decay_every=10000  --title-dim=250 --content-dim=250    --weight-decay=0 --type_='word' --debug-file='/tmp/debug'  --linear-hidden-size=2000 --zhuge=True  --augument=False

# 训练inception word
python2 main.py main --max_epoch=5 --plot_every=100 --env='inception-word' --weight=1 --model='CNNText_inception'  --zhuge=True --num-workers=4 --batch-size=512 --model-path=None --lr2=0 --lr=1e-3 --lr-decay=0.8  --decay-every=2500 --title-dim=1200 --content-dim=1200 --type_='word' --augument=False                                                   
# 训练inception char
python2 main.py main --max_epoch=5 --plot_every=100 --env='inception-char' --weight=1 --model='CNNText_inception'  --zhuge=True --num-workers=4 --batch-size=512 --model-path=None --lr2=0 --lr=1e-3 --lr-decay=0.8  --decay-every=2500 --title-dim=1200 --content-dim=1200 --type_='char'   --augument=False

# 训练FastText3 word
python2 main.py main --max_epoch=5 --plot_every=100 --env='fasttext3-word' --weight=5 --model='FastText3' --zhuge=True --num-workers=4 --batch-size=512  --lr2=1e-4 --lr=1e-3 --lr-decay=0.8  --decay-every=2500 --linear_hidden_size=2000 --type_='word'  --debug-file=/tmp/debugf --augument=False                           

```

大多数情况下,模型还能够通过finetune继续提升一定的分数,此时把学习率设为5e-5,再训练1-2个epoch左右即可,以LSTMText为例

```sh
python2 main.py main --max_epoch=2 --plot_every=100 --env='LSTMText-word-ft' --model='LSTMText'  --zhuge=True --num-workers=4 --batch-size=256 --model-path=None --lr2=5e-5 --lr=5e-5 --decay-every=5000 --type_='word'  --model-path='checkpoints/LSTMText_word_0.409196378421'                       
```

### 3.2 训练数据增强的模型

修改3.1脚本的`--augument=True`,训练即可.注意word的模型的分数会和之前的相匹敌,甚至还有提升,char模型分数会严重下降.同样在训练最后,还需要进行finetune.



### 3.3 各个模型的线下分数
根据本人的经验,线下分数会比线上低5-6个千分点,需要把这里的分数加上5-6个千分点才是线上真实的分数

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


作者能用上述模型融合得到至少0.433
### 3.4 训练MultiMoel
待续

- `models/MultiModelAll2`:主要思路是,把多个很优秀的模型,并行接在一起,然后接着训.一开始分数会下降的很厉害,然后慢慢的分数会开始上升
- `models/MultiModelAll`: 主要思路是,把多个很优秀的模型,并行接在一起,并且重新初始化他们的embedding为最初给的那个词向量,所有的子模型共享一个embedding.

MultiModel的分数,远小于这几个模型直接融合的分数,但是对于最终的融合有帮助.
MultiModel的训练,需要修改`main-all.py`或者`main-all.py`中的参数.主要设置
- opt.models: 指定MultiModel所包含的子模型
- opt.model_paths: 相对应的预训练好的模型保存路径

训练命令
```sh
python2 main-all.py main --max_epoch=2 --plot_every=10 --env='multimodelall-fast2' --weight=1 --model='MultiModelAll'  --zhuge=True --num-workers=4 --batch-size=128 --lr2=5e-4  --lr=1e-3 --lr-decay=0.5 --loss='bceloss'  --decay-every=4000 --all=True --debug-file=/tmp/debuga    
```



## 4 融合与提交csv
### 4.1 测试
根据上一步生成的多个最佳模型，对测试集进行测试，将测试结果保存成文件，用以融合,请注意修改模型保存的路径。测试的结果是测试集(或验证集)的所有样本属于各个类的概率. 如果是对测试集,那么生成的结果就是一个`217360*1999`的矩阵,如果是对验证集,生成的结果就是一个`200000*1999`的矩阵.

需要指定:

- model: 模型名字,包括`LSTMText`,`RCNN`,`MultiCNNTextBNDeep`,`FastText3`,`CNNText_inception`
- model-path: 模型保存的路径
- result-path: 结果保存路径
- val: 为True,会对验证集进行测试,为False会对测试集进行测试.

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

如果是对MultiModel进行测试,需要使用`test.3.py` ,并修改相对的`opt.models`和`opt.model_paths`,直接从`main-all.py`或者`mian-all.1.py`中copy过来即可.

比如
```sh
python2 test.3.py main --model='MultiModelAll'  --batch-size=128 --result-path='/data_ssd/zhihu/result/MultiModelallfast2_419245894992_aug_test.pth'  --val=False --zhuge=True --loss='bceloss'
```

### 4.2 融合
请参照`notebooks/val_ensemble.ipynb` 和`notebooks/test_ensemble.ipynb`,这个分别是验证集上融合计算分数,和测试集上融合生成相对应的csv.



### 5 文件说明
- `main.py`: 大多数模型的训练入口
- `config.py`: 配置文件,可通过命令喊参数传入覆盖默认值
- `test.1.py`: 根据模型生成对应的文件用来融合.
- `data/`: 数据加载相关
- `scripts/`: 各种脚本,主要是数据预处理
- `utils/` : 工具函数,包括计算分数,对可视化工具visdom的封装
- `models/`: 所有的模型定义
    - `models/BasicModel`: 所有模型的父类,封装实现了两个方法
    - `models/MultiCNNTextBNDeep`: CNN
    - `models/LSTMText`: RNN
    - `models/RCNN`: RCNN
    - `models/CNNText_inception` Inception
    - `models/MultiModelALL` 和`models/MultiModelAll2`
    -  其它模型,没用到,或者没效果    
- `rep.py`: 提交给主办方复现分数的代码
- `del/`: 失败的方法,没什么用的代码
- `notebooks/`: notebooks