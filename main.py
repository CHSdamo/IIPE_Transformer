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

    setting = '{}_sl{}_pl{}_bs{}_ep{}_el{}_dl{}_dim{}_heads{}'.format(args.model, args.seq_len, args.pred_len,
                                                                           args.batch_size, args.epochs,
                                                                           args.enc_layers, args.dec_layers,
                                                                           args.d_model, args.n_heads)
    exp = Experiment(args)
    print('>>>>>>>start training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(setting)

    # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp.test(setting)

    # if args.do_predict:
    #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #     exp.predict(setting, True)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = main_args()
    settings = parser.parse_args()

    main(settings)
