# coding: utf-8

# # seq2seq模型
import torch
from torch import nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()                    # priority selection gpu
device = torch.device('cuda' if use_cuda else 'cpu')

MAX_LENGTH = 10

# ## 1. 定义编码器
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        """
        vocab_size: the size of vocabulary
        hidden_size: the size of the RNN's hidden dimension
        n_layers: the number of the RNN layers, default is 1，note the defination of the layers is outside the 
                  RNN, so the numbers of the hidden units is (1*direction,N,hidden)!!!                 
        """
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        
        # embedding the words from one-hot to dense vector
        embedding_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # define the GRU unit
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)
        
    def forward(self, input_t, hidden_t):
        # (N,seq_size) => (N,seq_size,embedding_size)
        embedded = self.embedding(input_t)                        # [1,10]=>[1, 10, 128]
#         print(embedded.shape)
        for i in range(self.n_layers):                            # multi layer 
            embedded, hidden_t = self.gru(embedded, hidden_t)
        # embedded: (N,seq,embedding_size)  hidden_t: (1*direction, N, hidden)      
#         print(embedded.shape, hidden_t.shape)                   # torch.Size([1, 0, 128]) torch.Size([1, 1, 128])
        return embedded, hidden_t
    
    def initHidden(self):
        # (direction, batch, hidden) note batch is always 1，the way that gru is defined !!! 
        hidden_0 = torch.zeros(1, 1, self.hidden_size)
        return hidden_0

# 总结
# 
# - **batch_first参数只对输入和输出的batch和seq的顺序有影响，输入维度变为(batch,seq,feature)，输出维度(batch,seq,hidden*direction) **
# 
# **对隐藏状态的维度顺序无影响，仍旧是(layers*direction, batch,hidden)**，单向RNN的direction为1，双向为2
# - 上述编码器将输入维度(1,10)向量转换为(1,10,64)的向量，隐藏状态维度(1,1,64)
# - **上述方法中将layer层数定义在RNN外，这样只有一个隐藏状态，维度为(1*direction, batch,hidden)而定义在RNN内部的层数，最终隐藏状态维度为(layers*direction, batch,hidden)**

# ## 2. 定义解码器
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1):
        """
        hidden_size: the size of the hidden state
        vocab_size: the size of the vocabulary 
        n_layers: the number of the RNN layers
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        
        # embedding the word (N,1) => (N,1,embedding_size)
        embedding_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, batch_first=True) 
        self.linear  = nn.Linear(hidden_size, vocab_size)       # (N,1,vocab_size) output dims equal to vocabulary size
        self.softmax = nn.LogSoftmax(dim=2)
        
    def forward(self, input_t, hidden_t):
        embedded = self.embedding(input_t)                            # (N,1,embedding_size)
        for i in range(self.n_layers):
            embedded = F.relu(embedded)                               # ??? relu         
            embedded, hidden_t = self.gru(embedded, hidden_t)
#         print(embedded.shape)                                       # (N,1,embedding_size)
        output = self.softmax(self.linear(embedded))                  # (N,1,vocab_size)
        return output, hidden_t
    
    def initHidden(self):
        hidden_0 = torch.zeros([1, 1, self.hidden_size])              # give the state of the h0
        return hidden_0

# 总结：
# 
# 注意：解码器是对编码器生成的context隐藏向量进行解码，所以处理的句子长度为1，seq=1，**hidden维度大小为(batch,1,hidden)**
# 语言建模的句子长度？？？？？？

# ## 3. 定义有attention机制的解码器
class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        """
        hidden_size: the size of the hidden state
        vocab_size: the size of the vocabulary
        n_layers: the number of the RNN layers
        dropout_p: ???
        max_length: the maximum length of the sequence
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
#         self.dropout_p = dropout_p
        self.max_length = max_length
        
        # embedding the word (N,seq) => (N,seq,hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size) 
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)           # ??? why input_dim = hidden*2
        self.attn_combine = nn.Linear(self.hidden_size*2, self.hidden_size)  # ???
        self.dropout = nn.Dropout(dropout_p)                                 # ???
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)                        # # (N,seq,vocab_size)  softmax???
        
    def forward(self, input_t, hidden_t, encoder_ouputs):
        """
        input_t: (N, seq), N=1, seq=1
        hidden_t: (1*direction, N, hidden_size) , here we use unidirectional GRU=>direction=1
        encoder_ouputs:  (N,length, hidden) ???, equal to (N,seq,hidden)???
        
        """
        # embedding the word into dense vector, here embedding_size = hidden_size
        embedded = self.embedding(input_t)                  # get (N,seq,embedding_size)  N=1
        embedded = self.dropout(embedded)                   # ???
#         embedded = embedded.squeeze(1)  # batch, hidden   ???
        
        # compute the attention weight
        # embedded:(N,seq,hidden), here seq=1  |  hidden_t:(1,N,hidden)    ???seq不唯一怎么处理???
        embedded_cat = torch.cat([embedded, hidden_t.permute(1,0,2)], 2)   # get (N,1,2*hidden)
        attn_weights = F.softmax(self.attn(embedded_cat), dim=2)           # get (N,1,max_length)
        
        # (N,1,max_length)*(N,max_length,hidden) => (N,1,hidden)           batch multiply
        attn_applied = torch.matmul(attn_weights, encoder_ouputs)          # get (N,1,hidden)
        weight_out = torch.cat([embedded, attn_applied], 2)                # get (N,1,2*hidden)
        weight_out = self.attn_combine(weight_out)                         # get (N,1,hidden)
        
        # put data into gru
        for i in range(self.n_layers):
            weight = F.relu(weight_out)                                    # ??? 
            weight_out, hidden_t = self.gru(weight_out, hidden_t)
        
        output = F.log_softmax(self.out(weight_out), dim=2)                # get (N,1,vocab_size)
        return output, hidden_t, attn_weights
    
    def initHidden(self):
        hidden_0 = torch.zeros([1,1,self.hidden_size])
        return hidden_0


if __name__ == '__main__':
    # 1.test the encode model
    test_in = torch.randint(1,1000, [1,10]).long().to(device)        # input before Embedding must be LongTensor 
    print('1.Input tensor:', test_in.shape)
    encoder = EncoderRNN(1000, 128, 2).to(device)
    hidden_0 = encoder.initHidden().to(device)
    output, hidden = encoder(test_in, hidden_0)                      # output:(batch,seq,hidden*direction)
    print('2.outpu shape:', output.shape, 'hidden shape:', hidden.shape)

    # 2.test the decode model
    test_in = torch.randint(1,1000, [1,1]).long().to(device)        # input before Embedding must be LongTensor 
    print('1.Input tensor:', test_in.shape)
    decoder = DecoderRNN(1000, 128, 2).to(device)
    hidden_0 = decoder.initHidden().to(device)
    output, hidden = decoder(test_in, hidden_0)                      # output:(batch,seq,hidden*direction)
    print('2.output and hidden shape:', output.shape, hidden.shape)

    # test the decode model
    test_in = torch.randint(1,1000, [1,1]).long().to(device)        # input before Embedding must be LongTensor 
    print('1.Input tensor:', test_in.shape)
    decoder = AttnDecoderRNN(1000, 128, 2).to(device)
    hidden_0 = decoder.initHidden().to(device)
    encoder_outputs = torch.randn([1,MAX_LENGTH,128]).to(device)
    output, hidden, attn_weight = decoder(test_in, hidden_0, encoder_outputs)    # output:(batch,seq,hidden*direction)
    print('2.output and hidden shape:', output.shape, hidden.shape, attn_weight.shape)

