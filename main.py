import torch
import numpy as np
import random
from misc.args import main_args
from misc.utils import Experiment


def main(args):
    # set random seeds
    seed = args.seed
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda

    setting = '{}_sl{}_ll{}_pl{}_sd{}_bs{}_ep{}_el{}_dl{}_dim{}_heads{}_fc{}_attn{}_seed{}'.format(args.model,
                args.seq_len, args.label_len, args.pred_len, args.stride,
                args.batch_size, args.epochs, args.enc_layers, args.dec_layers, args.d_model, args.n_heads, args.d_ff,
                args.attn, args.seed)

    exp = Experiment(args)

    if args.model_mode == 'e2e' or args.model_mode == 'train':
        print('>>>>>>>start training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)

    if args.model_mode == 'e2e' or args.model_mode == 'pred':
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = main_args()
    settings = parser.parse_args()

    main(settings)
