# @Time    : 2019/4/17 23:36
# @Author  : shakespere
# @FileName: deep.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .cnn_clf import CNN_Classifier
from .lstm_clf import LSTM_Classifier
from .self_attention import SelfAttention
from .lstm_attention import LSTM_ATT
from .rcnn_clf import RCNN
from .config import LSTMTrainingConfig,CNNTrainingConfig,ATTTrainingConfig,LSTMATTTrainingConfig,RCNNTrainingConfig
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
class DeepModel(object):
    def __init__(self,vocab_size,embedding=None,method = "cnn"):
        assert method in ["cnn","lstm","self_att","lstm_att","rcnn"]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.method = method
        if method == "cnn":
            self.model = CNN_Classifier(vocab_size).to(self.device)
            self.epoches = CNNTrainingConfig.epoches
            self.learning_rate = CNNTrainingConfig.learning_rate
            self.print_step = CNNTrainingConfig.print_step
            self.lr_decay = CNNTrainingConfig.factor
            self.patience = CNNTrainingConfig.patience
            self.verbose = CNNTrainingConfig.verbose
        elif method == "lstm":
            self.model = LSTM_Classifier(vocab_size).to(self.device)
            self.epoches = LSTMTrainingConfig.epoches
            self.learning_rate = LSTMTrainingConfig.learning_rate
            self.print_step = LSTMTrainingConfig.print_step
            self.lr_decay = LSTMTrainingConfig.factor
            self.patience = LSTMTrainingConfig.patience
            self.verbose = LSTMTrainingConfig.verbose
        elif method == "self_att":
            self.batch_size = ATTTrainingConfig.batch_size
            self.model = SelfAttention(batch_size=ATTTrainingConfig.batch_size,vocab_size=vocab_size).cuda()#换GPU
            self.epoches = ATTTrainingConfig.epoches
            self.learning_rate = ATTTrainingConfig.learning_rate
            self.print_step = ATTTrainingConfig.print_step
            self.lr_decay = ATTTrainingConfig.factor
            self.patience = ATTTrainingConfig.patience
            self.verbose = ATTTrainingConfig.verbose
        elif method == "lstm_att":
            self.batch_size = LSTMATTTrainingConfig.batch_size
            self.model = LSTM_ATT(batch_size=LSTMATTTrainingConfig.batch_size,vocab_size=vocab_size).cuda()#需要使用GPU
            self.epoches = LSTMATTTrainingConfig.epoches
            self.learning_rate = LSTMATTTrainingConfig.learning_rate
            self.print_step = LSTMATTTrainingConfig.print_step
            self.lr_decay = LSTMATTTrainingConfig.factor
            self.patience = LSTMATTTrainingConfig.patience
            self.verbose = LSTMATTTrainingConfig.verbose
        elif method == "rcnn":
            self.batch_size = RCNNTrainingConfig.batch_size
            self.model = RCNN(batch_size=RCNNTrainingConfig.batch_size,vocab_size=vocab_size).cuda()#需要使用GPU
            self.epoches = RCNNTrainingConfig.epoches
            self.learning_rate = RCNNTrainingConfig.learning_rate
            self.print_step = RCNNTrainingConfig.print_step
            self.lr_decay = RCNNTrainingConfig.factor
            self.patience = RCNNTrainingConfig.patience
            self.verbose = RCNNTrainingConfig.verbose
        if embedding:
            self.model.init_embedding(embedding.to(self.device))
        self.optimizer = optim.Adam(
            self.model.parameters(),lr=self.learning_rate
        )
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=self.lr_decay,
            patience=self.patience,
            verbose=self.verbose
        )
        self.loss_fn = nn.BCELoss().to(self.device)
        self.best_acc = 0.
    def train_and_evel(self,train_loader,test_loader):
        """评估模型"""
        for epoch in range(1,self.epoches+1):
            print("Epoch {} training...".format(epoch))
            count = 0.
            step = 0
            losses = 0.
            train_acc = 0.
            train_correct_num = 0.
            for labels,sentences,lengths in train_loader:

                self.model.train()
                self.optimizer.zero_grad()
                if torch.cuda.is_available():
                    labels = labels.cuda()
                    sentences = sentences.cuda()

                #计算损失,update
                if self.method == "cnn":
                    probs = torch.sigmoid(self.model(sentences)).squeeze(1)#[B,]
                elif self.method == "lstm":
                    probs = torch.sigmoid(self.model(sentences,lengths)).squeeze(1) #[B,]
                elif self.method == "self_att":
                    probs = torch.sigmoid(self.model(sentences)).squeeze(1) #[B,]
                elif self.method == "lstm_att":
                    probs = torch.sigmoid(self.model(sentences)).squeeze(1) #[B,]
                elif self.method == "rcnn":
                    probs = torch.sigmoid(self.model(sentences)).squeeze(1) #[B,]
                loss = self.loss_fn(probs,labels.float())
                losses +=loss.item()
                loss.backward()
                clip_gradient(self.model,1e-1)#梯度裁剪
                self.optimizer.step()

                step+=1
                if step% self.print_step ==0:
                    print("Epoch {}:{}/{} {:.2f}% finished,Loss:{:.4f}".format(epoch,step,len(train_loader),
                                                                               100*step/len(train_loader),losses/self.print_step))
                    losses = 0
            self.test(test_loader)
        print("Best Accuracy:{:.2f}%".format(self.best_acc*100))
    def test(self,test_loader):
        """计算模型在测试集上的准确率以及损失"""
        count = 0.
        correct_num = 0.
        losses = 0.
        self.model.eval()
        with torch.no_grad():
            for labels,sentences,lengths in test_loader:
                labels = labels.to(self.device)
                sentences = sentences.to(self.device)

                if self.method == "cnn":
                    probs = torch.sigmoid(
                        self.model(sentences)
                    ).squeeze(1)#[B,]
                elif self.method == "lstm":
                    probs = torch.sigmoid(
                        self.model(sentences,lengths).squeeze(1) #[B,1]
                    )
                elif self.method == "self_att":
                    probs = torch.sigmoid(
                        self.model(sentences).squeeze(1) #[B,1]
                    )
                elif self.method == "lstm_att":
                    probs = torch.sigmoid(
                        self.model(sentences).squeeze(1) #[B,1]
                    )
                elif self.method == "rcnn":
                    probs = torch.sigmoid(
                        self.model(sentences).squeeze(1) #[B,1]
                    )
                loss = self.loss_fn(probs,labels.float())
                losses+=loss.item()
                pred_labels = torch.round(probs)#[B,]
                count +=len(labels)
                correct_num +=(pred_labels.long()==labels).sum().item()
        acc = correct_num/count
        if acc>self.best_acc:
            self.best_acc = acc
        print("Accuracy: {:.2f}%".format(100*acc))
        avg_loss = losses/len(test_loader)
        self.lr_scheduler.step(avg_loss)

def clip_gradient(model,clip_value):
    """
    梯度裁剪
    :param model:
    :param clip_value:
    :return:
    """
    params = list(filter(lambda p:p.grad is not None,model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value,clip_value)