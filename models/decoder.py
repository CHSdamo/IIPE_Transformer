import torch
import numpy as np
import torch.nn as nn
from models.embed import DataEmbedding
from models.attention import MultiHeadAttention
from models.encoder import PoswiseFeedForwardNet


class Decoder(nn.Module):
    def __init__(self, args, dataset):
        super(Decoder, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.token_dim = dataset.label_class_len()
        self.dec_embedding = DataEmbedding(args.d_model, dataset, args.dropout)
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.dec_layers)])  # Decoder的blocks

    def forward(self, dec_inputs, enc_outputs, batch_y_mark):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]   # 用在Encoder-Decoder Attention层
        """
        dec_outputs = self.dec_embedding(dec_inputs, batch_y_mark)  # [batch_size, tgt_len, d_model]

        # Masked Self_Attention：当前时刻是看不到未来的信息的
        dec_self_attn_mask = self.get_attn_mask(dec_inputs)  # [batch_size, tgt_len, tgt_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            # Decoder的Block是上一个Block的输出dec_outputs（变化）和Encoder网络的输出enc_outputs（固定）
            # dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attns, dec_enc_attns

    def get_attn_mask(self, seq):
        # seq: [batch_size, tgt_len]

        attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
        # attn_shape: [batch_size, tgt_len, tgt_len]
        attn_mask = np.triu(np.ones(attn_shape), k=1)   # 生成一个上三角矩阵 从index1开始[[0,1,1]]
        attn_mask = torch.from_numpy(attn_mask).byte().to(self.device)
        return attn_mask  # [batch_size, tgt_len, tgt_len]


class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(args)
        self.dec_enc_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask):  # , dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask)  # 这里的Q,K,V全是Decoder自己的输入
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs)
        # dec_enc_attn_mask)  # Attention层的Q(来自decoder) 和 K,V(来自encoder)
        dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn  # dec_self_attn, dec_enc_attn这两个是为了可视化的
