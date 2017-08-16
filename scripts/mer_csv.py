#coding:utf8
def merge(file1,file2,res='result.csv',k = 1):
    '''
    file1 准确率高 第一个位置的准确个数多
    file2 召回率高 所有位置的准确的个数多
    '''
    with open(file1) as f:
        lines1 = f.readlines()
    with open(file2) as f:
        lines2 = f.readlines()
    # import ipdb;
    # ipdb.set_trace()
    d1 = {_.split(',')[0]:_.strip().split(',')[1:] for _ in lines1}
    d2 = {_.split(',')[0]:_.strip().split(',')[1:] for _ in lines2}
    d = {}
    for key in d1:
        i1 = d1[key][:k]
        i2 = [_ for _ in d2[key] if _ not in i1][:(5-k)]
        d[key] = i1+i2
    
    lines = [[_] + _2 for _,_2 in d.iteritems()] 
    write_csv(res,lines)
    # rows=[0 for _ in range(result.shape[0])]
def merge2(file1,file2,res='result.csv',k = 1):
    '''
    file1 准确率高
    file2 召回率高
    '''
    with open(file1) as f:
        lines1 = f.readlines()
    with open(file2) as f:
        lines2 = f.readlines()
    # import ipdb;
    # ipdb.set_trace()
    d1 = {_.split(',')[0]:_.strip().split(',')[1:] for _ in lines1}
    d2 = {_.split(',')[0]:_.strip().split(',')[1:] for _ in lines2}
    d = {}
    for key in d1:
        _c1 = d1[key]
        _c2 = d2[key]
        i1 = d1[key][:k]
        i2 = [_ for _ in d2[key] if _ not in i1][:(5-k)]
        d[key] = i1+i2
    
    lines = [[_] + _2 for _,_2 in d.iteritems()] 
    write_csv(res,lines)

def write_csv(res,lines):
    import csv

    with open(res, "w") as f:
        csv_writer = csv.writer(f, dialect="excel")
        csv_writer.writerows(lines)


if __name__=='__main__':
    import fire
    fire.Fire()