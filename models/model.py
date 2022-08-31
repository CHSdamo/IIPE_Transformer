import torch
import torch.nn as nn
from models.encoder import Encoder, EncoderLayer
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, AttentionLayer
from models.embed import DataEmbedding
from misc.tools import Vocabulary


class Transformer(nn.Module):

    def __init__(self, dataset, d_model=512, n_heads=8, enc_layers=6, dec_layers=6, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=False, mix=True):

        super(Transformer, self).__init__()

        self.vocab_size = Vocabulary().vocab.__len__()
        self.output_attention = output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(d_model, dataset, dropout)
        self.dec_embedding = DataEmbedding(d_model, dataset, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                            FullAttention(False, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation) for l in range(enc_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True,  attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False,  attention_dropout=dropout, output_attention=True),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(dec_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        # self.projection = nn.Linear(d_model, c_out, bias=True)
        self.projection1 = nn.Linear(d_model, self.vocab_size, bias=False)
        self.projection2 = nn.Linear(d_model, 1)

    def forward(self, x_enc,  x_dec, x_mark_enc, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        dec_logits = self.projection1(dec_out)
        reg_output = self.projection2(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_logits, reg_output.squeeze(-1), attns
        else:
            return dec_logits, reg_output.squeeze(-1), None  # [B, L, D]

