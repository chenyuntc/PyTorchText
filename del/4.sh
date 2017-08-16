


# RCNN


python main.py main --max_epoch=2 --plot_every=100 --weight=1 --model='RCNN'  --env='rcnn-word' --batch-size=64  --lr=0.001 --lr2=0.000 --lr_decay=0.8 --decay_every=5000  --title-dim=500 --content-dim=500  --type_='word'  --kernel-size=2 --kmax-pooling=1 --linear-hidden-size=2000 --debug-file='/tmp/debugrcnn' --hidden-size=256 --decay_every=5000 --num-workers=4 --zhuge=True

