python2 main.py main --max_epoch=4 --plot_every=100 --env='lstm_char' --weight=1 --model='LSTMText'  --batch-size=128  --lr=0.001 --lr2=0 --lr_decay=0.5 --decay_every=10000  --type_='char'   --zhuge=True --linear-hidden-size=2000 --hidden-size=256 --kmax-pooling=3   --num-layers=3 --model-path='checkpoints/LSTMText_char'


python2 main.py main --max_epoch=4 --plot_every=100 --env='lstm_word' --weight=1 --model='LSTMText'  --batch-size=128  --lr=0.001 --lr2=0.0000 --lr_decay=0.5 --decay_every=10000  --type_='word'   --zhuge=True --linear-hidden-size=2000 --hidden-size=320 --kmax-pooling=2 --model-path='checkpoints/LSTMText_0726_13:38:49.pth'




python2 main.py main --max_epoch=4 --plot_every=100 --env='rcnn_char' --weight=1 --model='RCNN'  --batch-size=128  --lr=0.001 --lr2= --lr_decay=0.5 --decay_every=5000  --title-dim=1024 --content-dim=1024  --type_='char' --zhuge=True --kernel-size=3 --kmax-pooling=2 --linear-hidden-size=2000 --debug-file='/tmp/debugrcnn' --hidden-size=256 --num-layers=3
