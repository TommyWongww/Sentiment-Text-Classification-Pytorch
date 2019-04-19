# @Time    : 2019/4/17 21:47
# @Author  : shakespere
# @FileName: voc.py
from utils import load_stopwords
class Voc(object):
    """构建N-gram词典"""
    def __init__(self,N=1):
        self.N = N
        """加载禁用词集合"""
        self.stopwords = load_stopwords
        self.gram2id = {}
        self.id2gram = {}
        self.length = 0
        self.gram2count = {}
    def add_sentence(self,sentence):
        ngrams = [sentence[i:i+self.N] for i in range(len(sentence)-self.N-1)]
        ngrams = self.filter_stopgram(ngrams)
        for ngram in ngrams:
            self.add_gram(ngram)
    def __len__(self):
        return self.length
    def __str__(self):
        return "{}-Gram Voc(Length:{})".format(self.N,self.length) # 调试
    def __repr__(self):
        return str(self)
    def trim(self,min_count=3):
        """将出现次数少于min_count的词从字典中去掉"""
        keep_grams = []
        for gram,count in self.gram2count.items():
            if count>=min_count:
                keep_grams.append(gram)
        #重新构建词典
        self.gram2id = {}
        self.id2gram = {}
        self.length = 0
        for gram in keep_grams:
            self.add_gram(gram)
    def add_gram(self,ngram):
        """添加ngram到词典"""
        if ngram not in self.gram2id:
            self.gram2id[ngram] = self.length
            self.id2gram[self.length] = ngram
            self.gram2count[ngram] = 1
            self.length +=1
        else:
            self.gram2count[ngram] +=1
    def filter_stopgram(self,ngrams):
        """过滤禁用词"""
        filterd_ngrams = []
        for ngram in ngrams:
            """与禁用词表没有交集"""
            if not set(ngram).intersection(self.stopwords):
                filterd_ngrams.append(ngram)
        return filterd_ngrams