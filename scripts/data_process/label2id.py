#coding:utf8

'''
生成每个qid对应的label,保存成json
'''
def main(file,outfile):

    with open(file) as f:
        lines = f.readlines()

    def process(line):
        qid,labels = line.replace('\n','').split()
        labels = labels.split(',')
        return qid,labels
    results = list( map(process, lines))
    import ipdb;ipdb.set_trace()
    all_labels = { _ for ii,jj in results for _ in jj }
    sorted_labels = sorted(all_labels)
    label2id = {l_:ii for ii,l_ in enumerate(sorted_labels)}#-3239204820424->1
    id2label = {ii:l_ for ii,l_ in enumerate(sorted_labels)}

    d = {ii:[label2id[jj] for jj in labels ]  for ii,labels in results}

    data = dict(d=d,label2id=label2id,id2label=id2label)
    import json
    with open(outfile,'w') as f:
        json.dump(data,f) 

if __name__=='__main__':
    import fire
    fire.Fire()