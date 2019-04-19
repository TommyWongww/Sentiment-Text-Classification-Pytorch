# @Time    : 2019/4/17 22:14
# @Author  : shakespere
# @FileName: lr.py
import torch
from .config import LRConfig
class LogisticRegression(object):
    def __init__(self,input_size):
        self.learning_rate = LRConfig.learning_rate
        self.epoches = LRConfig.epoches
        self.input_size = input_size
        self.weights = torch.zeros(input_size+1)
    def train_and_evel(self,train_loader,test_loader):
        self.test(test_loader)
        for epoch in range(1,self.epoches+1):
            print("Epoch{} training.....".format(epoch))
            for labels,features in train_loader:
                labels = labels.float()
                features = features.float()
                exp_wx, features = self.forward(features)
                # 计算倒数
                gradient = features * labels.unsqueeze(1) - (features * exp_wx) / (1 + exp_wx)
                # [B,input_size+1]
                self.weights += self.learning_rate * torch.sum(gradient, dim=0)
            self.test(test_loader)
    def test(self,test_loader):
        count = 0.
        correct_num = 0.
        for labels,features in test_loader:
            labels = labels.float()
            features = features.float()
            exp_wx, features = self.forward(features)
            exp_wx = exp_wx.squeeze(1)
            pred_labels = torch.round(exp_wx / (1+exp_wx)).long()
            correct_num +=(pred_labels.float()==labels).sum().item()
            count+=len(labels)
        print("Accuracy: {:.4f}".format(correct_num / count))
    def forward(self,features):
        batch_size = features.size(0)
        bias = torch.ones(batch_size).unsqueeze(1)
        features = torch.cat([features,bias],dim=1)
        wx = torch.sum(features* self.weights,dim=1,keepdim=True) #[B,1]
        exp_wx = torch.exp(wx)
        return exp_wx,features


