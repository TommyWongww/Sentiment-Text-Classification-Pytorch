# @Time    : 2019/4/19 11:17
# @Author  : shakespere
# @FileName: lstm_attention.py
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from .config import LSTMATTConfig
class LSTM_ATT(nn.Module):
    def __init__(self,batch_size,vocab_size,weights=None):
        super(LSTM_ATT, self).__init__()
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.hidden_size = LSTMATTConfig.hidden_size
        self.weights = weights
        self.emb_size = LSTMATTConfig.emb_size
        self.embedding = nn.Embedding(vocab_size,self.emb_size)
        self.dropout = 0.8
        self.lstm = nn.LSTM(self.emb_size,self.hidden_size)
        self.label = nn.Linear(self.hidden_size,1)
        """可加如attn_layer"""
    def attention_net(self,lstm_output,final_state):
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output,hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights,1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1,2),soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state
    def forward(self,input,b_size=None):
        input = self.embedding(input)
        input = input.permute(1,0,2)
        if b_size == None:
            h_0 = Variable(torch.zeros(1,self.batch_size,self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(1,self.batch_size,self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(1,b_size,self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(1,b_size,self.hidden_size).cuda())
        output,(final_hidden_state,final_cell_state) = self.lstm(input,(h_0,c_0))# final_hidden_state.size() = (1, batch_size, hidden_size)
        output = output.permute(1,0,2)# output.size() = (batch_size, num_seq, hidden_size)
        attn_output = self.attention_net(output,final_hidden_state)
        logits = self.label(attn_output)
        return logits
    def init_embedding(self, embeddding):
        self.embedding.weight.data = embeddding