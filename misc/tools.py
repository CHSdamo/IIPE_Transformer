import torch


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class Vocabulary:
    def __init__(self):
        self.vocab = self.label2id()

    def __call__(self):
        len(self.vocab)

    def label2id(self):
        vocab = {'MABT1.7220.AC 000': 0,
                 'MABT1.7220.AC 001': 1,
                 'MABT1.7220.AC 002': 2,
                 'MABT1.7220.AC 003': 3,
                 'MABT1.7220.AC 004': 4,
                 'MABT1.7220.AC 005': 5,
                 'MABT1.7220.AC 006': 6,
                 'MABT1.7230.AC 000': 7,
                 'MABT1.7230.AC 001': 8,
                 'MABT1.7230.AC 002': 9,
                 'MABT1.7230.AC 003': 10,
                 'MABT1.7230.AC 004': 11,
                 'MABT1.7230.AC 005': 12,
                 'MABT1.7230.AC 006': 13,
                 'MABT1.7240.AC 000': 14,
                 'MABT1.7240.AC 001': 15,
                 'MABT1.7240.AC 002': 16,
                 'MABT1.7240.AC 003': 17,
                 'MABT1.7240.AC 004': 18,
                 'MABT1.7240.AC 005': 19,
                 'MABT1.7240.AC 006': 20,
                 'MABT1.7240.AC 007': 21,
                 'MABT1.7240.AC 008': 22,
                 'MABT1.7240.AC 009': 23,
                 'MABT1.7240.AC 010': 24,
                 'MABT1.7240.AC 011': 25,
                 'MABT1.7240.AC 012': 26,
                 'MABT1.7240.AC 013': 27,
                 'MABT1.7240.AC 014': 28,
                 'MABT1.7240.AC 015': 29,
                 'MABT1.7240.AC 016': 30,
                 'MABT1.7240.AC 017': 31,
                 'MABT1.7240.AC 018': 32,
                 'MABT1.7240.AC 019': 33,
                 'MABT1.7240.AC 020': 34,
                 'MABT1.7240.AC 021': 35,
                 'MABT1.7240.AC 022': 36,
                 'MABT1.7240.AC 023': 37,
                 'MABT1.7240.AC 024': 38}

        return vocab

    def id2label(self):
        idx2label = {i: w for i, w in enumerate(self.vocab)}
        return idx2label

    def car2label(self):

        return

'''
'MABT1.7240.0': 0,
                 'MABT1.7240.1': 1,
                 'MABT1.7240.2': 2,
                 'MABT1.7240.3': 3,
                 'MABT1.7240.4': 4,
                 'MABT1.7240.5': 5,
                 'MABT1.7240.6': 6,
                 'MABT1.7240.7': 7,
                 'MABT1.7240.8': 8,
                 'MABT1.7240.9': 9,
                 'MABT1.7240.10': 10,
                 'MABT1.7240.11': 11,
                 'MABT1.7240.12': 12,
                 'MABT1.7240.13': 13,
                 'MABT1.7240.14': 14,
                 'MABT1.7240.15': 15,
                 'MABT1.7240.16': 16,
                 'MABT1.7240.17': 17,
                 'MABT1.7240.18': 18,
                 'MABT1.7240.19': 19,
                 'MABT1.7240.20': 20,
                 'MABT1.7240.21': 21,
                 'MABT1.7240.22': 22,
                 'MABT1.7240.23': 23,
                 'MABT1.7240.24': 24

'''





'''
                 'MABT1.7220.0': 0,
                 'MABT1.7220.1': 1,
                 'MABT1.7220.2': 2,
                 'MABT1.7220.3': 3,
                 'MABT1.7220.4': 4,
                 'MABT1.7220.5': 5,
                 'MABT1.7220.6': 6,
                 'MABT1.7230.0': 7,
                 'MABT1.7230.1': 8,
                 'MABT1.7230.2': 9,
                 'MABT1.7230.3': 10,
                 'MABT1.7230.4': 11,
                 'MABT1.7230.5': 12,
                 'MABT1.7230.6': 13,
                 'MABT1.7240.0': 14,
                 'MABT1.7240.1': 15,
                 'MABT1.7240.2': 16,
                 'MABT1.7240.3': 17,
                 'MABT1.7240.4': 18,
                 'MABT1.7240.5': 19,
                 'MABT1.7240.6': 20,
                 'MABT1.7240.7': 21,
                 'MABT1.7240.8': 22,
                 'MABT1.7240.9': 23,
                 'MABT1.7240.10': 24,
                 'MABT1.7240.11': 25,
                 'MABT1.7240.12': 26,
                 'MABT1.7240.13': 27,
                 'MABT1.7240.14': 28,
                 'MABT1.7240.15': 29,
                 'MABT1.7240.16': 30,
                 'MABT1.7240.17': 31,
                 'MABT1.7240.18': 32,
                 'MABT1.7240.19': 33,
                 'MABT1.7240.20': 34,
                 'MABT1.7240.21': 35,
                 'MABT1.7240.22': 36,
                 'MABT1.7240.23': 37,
                 'MABT1.7240.24': 38
'''



'''
                 'MABT2.7260.0': 39,
                 'MABT2.7260.1': 40,
                 'MABT2.7260.2': 41,
                 'MABT2.7260.3': 42,
                 'MABT2.7260.4': 43,
                 'MABT2.7260.5': 44,
                 'MABT2.7260.6': 45,
                 'MABT2.7260.7': 46,
                 'MABT2.7260.8': 47,
                 'MABT2.7260.9': 48,
                 'MABT2.7260.10': 49,
                 'MABT2.7260.11': 50,
                 'MABT2.7260.12': 51,
                 'MABT2.7260.13': 52,
                 'MABT2.7260.14': 53,
                 'MABT2.7260.15': 54,
                 'MABT2.7260.16': 55,
                 'MABT2.7260.17': 56,
                 'MABT2.7260.18': 57,
                 'MABT2.7260.19': 58,
                 'MABT2.7260.20': 59,
                 'MABT2.7260.21': 60,
                 'MABT2.7260.22': 61,
                 'MABT2.7260.23': 62,
                 'MABT2.7260.24': 63,
                 'MABT2.7260.25': 64,
                 'MABT2.7260.26': 65,
                 'MABT2.7260.27': 66,
                 'MABT2.7260.28': 67,
                 'MABT2.7260.29': 68,
                 'MABT2.7260.30': 69,
                 'MABT2.7260.31': 70,
                 'MABT2.7260.32': 71,
                 'MABT2.7260.33': 72,
                 'MABT2.7260.34': 73,
                 'MABT2.7260.35': 74,
                 'MABT2.7260.36': 75,
                 'MABT2.7260.37': 76,
                 'MABT2.7260.38': 77,
                 'MABT2.7260.39': 78,
                 'MABT2.7260.40': 79,
                 'MABT2.7260.41': 80,
                 'MABT2.7260.42': 81,
                 'MABT2.7260.43': 82,
                 'MABT2.7260.44': 83,
                 'MABT3.7280.0': 84,
                 'MABT3.7280.1': 85,
                 'MABT3.7280.2': 86,
                 'MABT3.7280.3': 87,
                 'MABT3.7280.4': 88,
                 'MABT3.7280.5': 89,
                 'MABT3.7280.6': 90,
                 'MABT3.7280.7': 91,
                 'MABT3.7280.8': 92,
                 'MABT3.7280.9': 93,
                 'MABT3.7280.10': 94,
                 'MABT3.7280.11': 95,
                 'MABT3.7280.12': 96,
                 'MABT3.7280.13': 97,
                 'MABT3.7280.14': 98,
                 'MABT3.7280.15': 99,
                 'MABT3.7280.16': 100,
                 'MABT3.7280.17': 101,
                 'MABT3.7280.18': 102,
                 'MABT3.7280.19': 103,
                 'MABT3.7280.20': 104,
                 'MABT3.7280.21': 105,
                 'MABT3.7280.22': 106,
                 'MABT3.7280.23': 107,
                 'MABT3.7280.24': 108,
                 'MABT3.7280.25': 109,
                 'MABT3.7280.26': 110,
                 'MABT3.7280.27': 111,
                 'MABT3.7280.28': 112,
                 'MABT3.7280.29': 113,
                 'MABT3.7280.30': 114,
                 'MABT3.7280.31': 115,
                 'MABT3.7280.32': 116,
                 'MABT3.7280.33': 117,
                 'MABT3.7280.34': 118,
                 'MABT3.7280.35': 119,
                 'MABT3.7280.36': 120,
                 'MABT3.7280.37': 121,
                 'MABT3.7280.38': 122,
                 'MABT3.7280.39': 123,
                 'MABT3.7280.40': 124,
                 'MABT3.7280.41': 125,
                 'MABT3.7280.42': 126,
                 'MABT3.7280.43': 127
'''