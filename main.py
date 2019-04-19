# @Time    : 2019/4/17 22:02
# @Author  : shakespere
# @FileName: main.py
from functools import partial
from torch.utils.data import DataLoader,Dataset
from data import DPDataSet
from utils import collate_fn_dl,collate_fn_ml,load_word2id,load_embeddings
from models.deep import DeepModel
from models.lr import LogisticRegression
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
VOCAB_SIZE = 3500#指字典大小
SENT_MAX_LEN = 128 #指定句子最长的长度
batch_size = 128
def main():
    """在训练集上构建一元词典和二元词典"""
    word2id = load_word2id(length=VOCAB_SIZE)
    """prepare dataset"""
    train_loader = DataLoader(
        dataset=DPDataSet('train'),
        batch_size=batch_size,
        collate_fn=partial(collate_fn_dl,word2id,SENT_MAX_LEN),
        drop_last=True,
        pin_memory=True,
        # num_workers=4,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=DPDataSet("test"),
        batch_size=batch_size,
        collate_fn=partial(collate_fn_dl,word2id,SENT_MAX_LEN),
        pin_memory=True,drop_last=True,shuffle=True
    )
    vocab_size = len(word2id)
    print("Vocab Size:",vocab_size)
    print("加载词向量.....")
    try:
        embedding = load_embeddings(word2id)
    except FileNotFoundError:
        embedding = None
    print("测试BiLSTM:")
    lstm_model = DeepModel(vocab_size, embedding, method="lstm")
    lstm_model.train_and_evel(train_loader, test_loader)

    print("测试CNN:")
    cnn_model = DeepModel(vocab_size,embedding,method="cnn")
    cnn_model.train_and_evel(train_loader,test_loader)

    print("测试selfAttention:")
    att_model = DeepModel(vocab_size,embedding,method="self_att")
    att_model.train_and_evel(train_loader,test_loader)

    print("测试LSTM_Attention:")
    lstm_att_model = DeepModel(vocab_size, embedding, method="lstm_att")
    lstm_att_model.train_and_evel(train_loader, test_loader)
    print("测试RCNN:")
    RCNN_model = DeepModel(vocab_size, embedding, method="rcnn")
    RCNN_model.train_and_evel(train_loader, test_loader)

    # ##机器学习算法
    # train_loader_ml = DataLoader(
    #     dataset=DPDataSet('train'),
    #     batch_size=64,
    #     collate_fn=partial(collate_fn_ml,word2id)
    # )
    # test_loader_ml = DataLoader(
    #     dataset=DPDataSet('test'),
    #     batch_size=64,
    #     collate_fn = partial(collate_fn_ml,word2id)
    # )
    # print("使用LR模型进行分类...")
    # input_size = len(word2id)
    # lr_model = LogisticRegression(input_size)
    # lr_model.train_and_evel(train_loader_ml,test_loader_ml)

if __name__ == "__main__":
    main()
