# encoding: utf-8
'''
Word2Vec模型实现
'''
from gensim.models import Word2Vec
import gensim
import re, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def sentence2words(sentence, stopWords=False, stopWords_set=None):
    """
    split a sentence into words based on jieba
    """
    return sentence.split(',')

class MySentences(object):
    def __init__(self, list_csv):
        with open(list_csv, 'r') as f:
            self.fns = [line.strip() for line in f]
    def __iter__(self):
        for line in self.fns:
            content = line#.replace("-","").replace(" ","")
            if len(content) != 0:
                yield sentence2words(content.strip())

    def train_save(self, list_csv):
        sentences = MySentences(list_csv)
        num_features = 256
        min_word_count = 1
        num_workers = 20
        context = 5
        epoch = 20
        sample = 1e-5
        model = Word2Vec(
            sentences,
            size=num_features,
            min_count=min_word_count,
            workers=num_workers,
            sample=sample,
            window=context,
            iter=epoch,
        )
        #model.save(model_fn)
        return model

if __name__ == "__main__":
    ms = MySentences('../ddd/topic_graph.txt')
    model = ms.train_save('../ddd/topic_graph.txt')
    model.save('../ddd/topic_vec')
    output=open(r'../ddd/topic_vec.txt','a+')
    model.wv.save_word2vec_format(output, binary=False)
