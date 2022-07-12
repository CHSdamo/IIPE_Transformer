import torch.nn as nn
from models.embed import DataEmbedding
from models.attention import MultiHeadAttention


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.enc_embedding = DataEmbedding(args.d_model, args.dropout)
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.enc_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """
        enc_outputs = self.enc_embedding(enc_inputs)

        # Encoder输入序列的pad mask矩阵
        '''
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        '''

        enc_self_attns = []  # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系
        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            '''
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention            
            '''
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)  # 这个只是为了可视化
        return enc_outputs, enc_self_attns


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, enc_inputs):
    # def forward(self, enc_inputs, enc_self_attn_mask):
        """E
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        '''
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) 
        '''
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V（未线性变换前）
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.args = args
        self.fc = nn.Sequential(
            nn.Linear(args.d_model, args.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(args.d_ff, args.d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.args.d_model)(output + residual)  # [batch_size, seq_len, d_model]
