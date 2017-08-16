#coding:utf8

'''
将embedding.txt 转成numpy矩阵
'''


import word2vec
import numpy as np

def main(em_file, em_result):
    '''
    embedding ->numpy
    '''
    em = word2vec.load(em_file)
    vec = (em.vectors)
    word2id = em.vocab_hash
    # d = dict(vector = vec, word2id = word2id)
    # t.save(d,em_result)
    np.savez_compressed(em_result,vector=vec,word2id=word2id)

if __name__ == '__main__':
    import fire
    fire.Fire()