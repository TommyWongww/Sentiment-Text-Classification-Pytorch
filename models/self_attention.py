# @Time    : 2019/4/18 13:37
# @Author  : shakespere
# @FileName: self_attention.py
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from .config import ATTConfig
class SelfAttention(nn.Module):
    def __init__(self,batch_size,vocab_size,weights=None):
        super(SelfAttention, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = ATTConfig.hidden_size
        self.vocab_size = vocab_size
        self.emb_size = ATTConfig.emb_size
        self.weights = weights
        self.embedding = nn.Embedding(vocab_size,self.emb_size)
        # self.embedding.weights = nn.Parameter(weights,requires_grad=False)
        self.dropout = 0.8
        self.bilstm = nn.LSTM(self.emb_size,self.hidden_size,dropout=self.dropout,bidirectional=True)
        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
        self.W_s1 = nn.Linear(2*self.hidden_size,350)
        self.W_s2 = nn.Linear(350,30)
        self.fc_layer = nn.Linear(30*2*self.hidden_size,2000)
        self.label = nn.Linear(2000,1)
    def attention_net(self,lstm_output):
        attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0,2,1)
        attn_weight_matrix = F.softmax(attn_weight_matrix,dim=2)
        return attn_weight_matrix
    def forward(self,input_sentences,batch_size=None):
        input = self.embedding(input_sentences)
        input = input.permute(1,0,2)
        if batch_size is None:
            h_0 = Variable(torch.zeros(2,self.batch_size,self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(2,self.batch_size,self.hidden_size).cuda())
        else:
            h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size).cuda())
        output,(h_n,c_n) = self.bilstm(input,(h_0,c_0))
        output = output.permute(1,0,2)
        attn_weight_matrix = self.attention_net(output)
        hidden_matrix = torch.bmm(attn_weight_matrix,output)
        fc_out = self.fc_layer(hidden_matrix.view(-1,hidden_matrix.size()[1]*hidden_matrix.size()[2]))
        logits = self.label(fc_out)
        return logits
    def init_embedding(self,embeddding):
        self.embedding.weight.data = embeddding
