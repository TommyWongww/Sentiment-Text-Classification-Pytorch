# @Time    : 2019/4/17 20:08
# @Author  : shakespere
# @FileName: utils.py
#-*- coding : utf-8 -*-
# coding:utf-8
import string
import torch
def load_stopwords(path = "./data/stopwords.txt"):
    with open(path,"r") as f:
        stop_words = [word.strip('\n') for word in f]
    stop_words +=list(string.printable)
    return set(stop_words)
def load_word2id(length=2000,vocab_path="./data/vocab.csv"):
    word2id = {"<pad>":0,"<unk>":1}
    with open(vocab_path,"r") as f:
        words = [line.split(',')[0] for line in f]
    for word in words[:length]:
        word2id[word] = len(word2id)
    return word2id
def load_embeddings(word2id,emb_dim=300,emb_path = "./data/pre"):
    vocab_size = len(word2id)
    embedding = torch.Tensor(vocab_size,emb_dim)
    word2embstr = {}
    with open(emb_path,"r") as f:
        for line in f:
            word,embstr = line.split(' ',1)
            word2embstr[word] = embstr.strip("\n")
    #find the embedding that we need
    for word,word_id in word2id.items():
        if word in word2embstr:
            embs = list(map(float,word2embstr[word].split()))
            embedding[word_id] = torch.Tensor(embs)
        else:
            embedding[word_id] = torch.randn(emb_dim)
    print("building embedding finished...")
    return embedding
def collate_fn_ml(word2id,batch):
    """为ML分类方法提供数据，将文本转化为向量"""
    labels,sentences = zip(*batch)
    labels = torch.LongTensor(labels)
    bsize = len(sentences)
    length = len(word2id)
    sent_tensor = torch.zeros(bsize,length).long()
    for sent_id,sent in enumerate(sentences):
        for gram in sent:
            if gram in word2id:
                gram_id = word2id[gram]
                sent_tensor[sent_id][gram_id] +=1
    return labels,sent_tensor
def collate_fn_dl(word2id,max_len,batch):
    """为DL分类方法提供数据，根据句子长度进行排序"""
    batch.sort(key=lambda pair:len(pair[1]),reverse=True)
    labels,sentences = zip(*batch)
    #截断，取前64个字
    sentences = [sent[:64] for sent in sentences]
    labels = torch.LongTensor(labels)
    pad_id = word2id["<pad>"]
    unk_id = word2id["<unk>"]
    bsize = len(sentences)
    max_len=max(len(sentences[0]),max_len)
    sent_tensor = torch.ones(bsize,max_len).long()*pad_id
    for sent_id,sent in enumerate(sentences):
        for word_id,word in enumerate(sent):
            sent_tensor[sent_id][word_id] = word2id.get(word,unk_id)
    lengths = [len(sent) for sent in sentences]
    return labels,sent_tensor,lengths
def prepropress_for_ml(sentences):
    """将字与字之间用空格隔开分词"""
    sentences = [" ".join(list(sent)) for sent in sentences]
    #可加入二维特征
    return sentences
def get_feature(sent):
    """
    抽取1-gram以及2-gram特征
    :param sent:
    :return:
    """
    unigrams = list(sent)
    bigrams = [sent[i:i+2] for i in range(len(sent)-1)]
    return unigrams+bigrams
