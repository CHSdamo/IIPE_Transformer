import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder
from dataset.vocab import Vocabulary


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        # self.args = args

        self.vocab_size = Vocabulary().vocab_size
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.projection1 = nn.Linear(args.d_model, self.vocab_size, bias=False)
        self.projection2 = nn.Linear(args.d_model, 1)

    def forward(self, enc_inputs, dec_inputs):
        """Transformers的输入：两个序列
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        """
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # 经过Encoder网络后，得到的输出还是[batch_size, src_len, d_model]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # dec_outputs: [batch_size, tgt_len, d_model],
        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
        # dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # dec_outputs: [batch_size, tgt_len, d_model] -> dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection1(dec_outputs)
        reg_output = self.projection2(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), reg_output.squeeze(-1), enc_self_attns, dec_self_attns, dec_enc_attns
