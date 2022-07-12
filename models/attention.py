import torch.nn as nn
import torch
import numpy as np


class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    输入：seq_len x d_model
    输出：seq_len x d_model
    """
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        self.W_Q = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(args.d_model, args.d_k * args.n_heads, bias=False)
        self.W_V = nn.Linear(args.d_model, args.d_v * args.n_heads, bias=False)
        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        self.fc = nn.Linear(args.n_heads * args.d_v, args.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换               拆成多头

        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.args.n_heads, self.args.d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1, self.args.n_heads, self.args.d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, self.args.n_heads, self.args.d_v).transpose(1, 2)

        # 因为是多头，所以mask矩阵要扩充成4维的
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.args)(Q, K, V, attn_mask)
        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.args.n_heads * self.args.d_v)

        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.args.d_model)(output + residual), attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, args):
        super(ScaledDotProductAttention, self).__init__()
        self.args = args

    def forward(self, Q, K, V, attn_mask=None):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.args.d_k)

        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        if attn_mask is not None:
            scores.masked_fill_(attn_mask.bool(), -1e9)  # Fills elements of self tensor with value where mask is True.
        # softmax 时，e^0=1, e^-inf = 0
        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax  ,即 alpha
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        context = torch.matmul(attn, V)  # context: [batch_size, n_heads, len_q, d_v]
        # context：[[z1,z2,...],[...]]向量, attn注意力稀疏矩阵（用于可视化的）
        return context, attn
