import argparse


def main_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=30, help="seed to use: [default=30]")
    parser.add_argument('--folder', type=str, required=True)

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--seq_len', type=int, default=7, help='input sequence length of Informer encoder')
    parser.add_argument('--pred_len', type=int, default=7, help='prediction sequence length')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', type=str, default='transformer')

    '''
    Transformer Parameters
    '''
    parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--enc_layers', type=int, default=6, help='num of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=6, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='optimizer learning rate')
    parser.add_argument('--optim', type=str, default='adam')

    return parser
