# @Time    : 2019/4/19 11:55
# @Author  : shakespere
# @FileName: rcnn_clf.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from .config import RCNNConfig
class RCNN(nn.Module):
    def __init__(self,batch_size,vocab_size,weights=None):
        super(RCNN, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden_size = RCNNConfig.hidden_size
        self.emb_size = RCNNConfig.emb_size
        self.embedding = nn.Embedding(vocab_size,self.emb_size)
        self.dropout = 0.8
        self.lstm = nn.LSTM(self.emb_size,self.hidden_size,dropout=self.dropout,bidirectional=True)
        self.W2 = nn.Linear(2*self.hidden_size+self.emb_size,self.hidden_size)
        self.label = nn.Linear(self.hidden_size,1)#1更换为想输出的类别即可
    def forward(self,input,b_size = None):
        input = self.embedding(input)# embedded input of shape = (batch_size, num_sequences, embedding_length)
        input = input.permute(1,0,2)# input.size() = (num_sequences, batch_size, embedding_length)
        if b_size is None:
            h_0 = Variable(torch.zeros(2,self.batch_size,self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(2,self.batch_size,self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(2, b_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(2, b_size, self.hidden_size).cuda())
        output,(final_hidden_state,final_cell_state) = self.lstm(input,(h_0,c_0))
        final_encoding = torch.cat((output,input),2)
        final_encoding = final_encoding.permute(1,0,2)
        y = self.W2(final_encoding)# y.size() = (batch_size, num_sequences, hidden_size)
        y = y.permute(0,2,1)# y.size() = (batch_size, hidden_size, num_sequences)
        y = F.max_pool1d(y,y.size()[2]) # y.size() = (batch_size, hidden_size, 1)
        y = y.squeeze(2)
        logits = self.label(y)
        return logits
    def init_embedding(self,embeddding):
        self.embedding.weight.data = embeddding
