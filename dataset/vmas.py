import os

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
# from sklearn.preprocessing import *
import pandas as pd
import logging
import tqdm
import numpy as np
import torch
from dataset.vocab import Vocabulary


class VMASDataset(Dataset):
    def __init__(self, args):

        self.encoded_df = None
        self.encoder_fit = {}
        self.dataset_len = None
        self.data = []
        self.pred = []
        self.labels = []
        self.targets = []
        self.window_label = []
        # self.vocab = self.label2id()
        self.args = args
        self.raw_df = self.read_csv_file()
        self.processed_df = self.preprocess()
        self.encode_data()
        self.prepare_samples()
        # self.split_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = torch.tensor(self.data[index])
        pred = torch.tensor(self.pred[index])
        # labels = torch.tensor(self.labels[index])
        # targets = torch.tensor(self.targets[index])
        return sample, pred   # , labels, targets



    @staticmethod
    def encoder_fit_transform(column, enc_type="label"):
        if enc_type == "label":
            enc = LabelEncoder()
        else:
            enc = StandardScaler()
        enc.fit(column)

        return enc.transform(column)

    def prepare_samples(self):
        if not os.path.exists('./saves/trans_data.npy'):
            trans_data, trans_label, trans_target = [], [], []

            for _, row in self.processed_df.iterrows():
                row = list(row)

                row[-2] = Vocabulary().label2id()[row[-2]]
                trans_data.append(row)
                trans_label.append(row[-2])
                trans_target.append(row[-1])

            np.save('./saves/trans_data.npy', np.array(trans_data))
            np.save('./saves/trans_label.npy', np.array(trans_label))
            np.save('./saves/trans_target.npy', np.array(trans_target))
        else:
            trans_data = np.load('./saves/trans_data.npy').tolist()
            trans_label = np.load('./saves/trans_label.npy').tolist()
            trans_target = np.load('./saves/trans_target.npy').tolist()

        for i in range(0, len(trans_data) - self.args.seq_len - self.args.pred_len, self.args.stride):
            sample = trans_data[i:i + self.args.seq_len]
            pred = trans_data[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len]
            label = trans_label[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len]
            target = trans_target[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len]

            self.data.append(sample)
            self.pred.append(pred)
            self.labels.append(label)
            self.targets.append(target)

        assert len(self.data) ==len(self.pred) == len(self.labels) == len(self.targets)

    def encode_data(self):
        self.dataset_len = len(self.processed_df)
        # sub_columns = ['Code_x', 'Area', 'Station', 'Section', 'UniqueID']
        # col_data = list(self.processed_df['Label'])
        # # for _, token in enumerate(col_data):
        #
        #
        # for col_name in tqdm.tqdm(sub_columns, desc='Encoding data:'):
        #     col_data = self.raw_df[col_name]
        #     col_fit, col_data = self.encoder_fit_transform(col_data)
        #     self.encoder_fit[col_name] = col_fit
        #     self.raw_df[col_name] = col_data

        col_data = self.encoder_fit_transform(self.processed_df[['Car Code']])
        self.processed_df['Car Code'] = col_data
        col_data = self.encoder_fit_transform(self.processed_df[['Duration']], enc_type='value')
        self.processed_df['Duration'] = col_data

        # selected_columns = ['UniqueID', 'Code_x', 'Action', 'Area', 'Station', 'Section', 'Error Types', 'Time_diff(s)']
        #
        # self.encoded_df = self.raw_df[selected_columns]

    def read_csv_file(self):
        file = './saves/merged_df.csv'
        if not os.path.exists(file):
            error_type_df = pd.read_csv(self.args.folder + '7carCodes_ACT_All.csv')
            # read tsv file, seperate with Space
            ev_zip_df = pd.read_csv(self.args.folder + 'Export_LEGATO_20210111_1021_20210517_0559.evzip.tsv', sep='\t')
            error_type_df.rename(
                columns={'Event': 'Action', 'Fehlertyp': 'Error Types', 'Fehler Anzahl': 'Error Counts',
                         'Time_diff(s)': 'Duration', 'Code': 'Car Code'},
                inplace=True)  # with error types
            ev_zip_without_ev_df = ev_zip_df[ev_zip_df['Event'].str.contains('AC')]  # del EV no ev

            join = pd.merge(error_type_df, ev_zip_without_ev_df, on='Timestamp', how='left')
            # del duplicate:
            join = join[join['Duration'] == join['x1'].apply(lambda x: float(x.replace(',', '.'))).astype(float)]
            merged_df = join[
                join['Action'] == join['Event'].str.replace('AC', '').astype(int)]  # del dup in action & event

            os.makedirs('./saves', exist_ok=True)
            merged_df.to_csv(file, index=False)
        else:
            merged_df = pd.read_csv(file)
        return merged_df

    def preprocess(self):
        file = './saves/processed_df.csv'
        if not os.path.exists(file):
            # delete whole seq with Time_diff>600 component:
            df = self.raw_df[
                ~self.raw_df['UniqueID'].isin(self.raw_df[self.raw_df['Duration'] >= 600]['UniqueID'].unique())] #(617159)

            df.reset_index(inplace=True)
            df['Label'] = df[['Section', 'Station', 'Action']].astype(str).agg('.'.join, axis=1)
            # df.drop(['Error Counts', 'Event', 'Unnamed: 6', 'Unnamed: 7', 'Code_y', 'x1', 'index', 'Section',
            #          'Station', 'Action', 'Timestamp', 'Area', 'UniqueID', 'Unnamed: 0'], axis=1, inplace=True)
            # df['UniqueID'] = np.floor(pd.to_numeric(join['UniqueID'], errors='coerce')).astype('Int64')
            sub_col = ['Car Code', 'Label', 'Duration']
            processed_df = df[sub_col]
            os.makedirs('./saves', exist_ok=True)
            processed_df.to_csv(file, index=False)
        else:
            processed_df = pd.read_csv(file)
        return processed_df

    # def label2id(self):
    #     vocab = {'MABT1.7220.0': 0,
    #              'MABT1.7220.1': 1,
    #              'MABT1.7220.2': 2,
    #              'MABT1.7220.3': 3,
    #              'MABT1.7220.4': 4,
    #              'MABT1.7220.5': 5,
    #              'MABT1.7220.6': 6,
    #              'MABT1.7230.0': 7,
    #              'MABT1.7230.1': 8,
    #              'MABT1.7230.2': 9,
    #              'MABT1.7230.3': 10,
    #              'MABT1.7230.4': 11,
    #              'MABT1.7230.5': 12,
    #              'MABT1.7230.6': 13,
    #              'MABT1.7240.0': 14,
    #              'MABT1.7240.1': 15,
    #              'MABT1.7240.2': 16,
    #              'MABT1.7240.3': 17,
    #              'MABT1.7240.4': 18,
    #              'MABT1.7240.5': 19,
    #              'MABT1.7240.6': 20,
    #              'MABT1.7240.7': 21,
    #              'MABT1.7240.8': 22,
    #              'MABT1.7240.9': 23,
    #              'MABT1.7240.10': 24,
    #              'MABT1.7240.11': 25,
    #              'MABT1.7240.12': 26,
    #              'MABT1.7240.13': 27,
    #              'MABT1.7240.14': 28,
    #              'MABT1.7240.15': 29,
    #              'MABT1.7240.16': 30,
    #              'MABT1.7240.17': 31,
    #              'MABT1.7240.18': 32,
    #              'MABT1.7240.19': 33,
    #              'MABT1.7240.20': 34,
    #              'MABT1.7240.21': 35,
    #              'MABT1.7240.22': 36,
    #              'MABT1.7240.23': 37,
    #              'MABT1.7240.24': 38,
    #              'MABT2.7260.0': 39,
    #              'MABT2.7260.1': 40,
    #              'MABT2.7260.2': 41,
    #              'MABT2.7260.3': 42,
    #              'MABT2.7260.4': 43,
    #              'MABT2.7260.5': 44,
    #              'MABT2.7260.6': 45,
    #              'MABT2.7260.7': 46,
    #              'MABT2.7260.8': 47,
    #              'MABT2.7260.9': 48,
    #              'MABT2.7260.10': 49,
    #              'MABT2.7260.11': 50,
    #              'MABT2.7260.12': 51,
    #              'MABT2.7260.13': 52,
    #              'MABT2.7260.14': 53,
    #              'MABT2.7260.15': 54,
    #              'MABT2.7260.16': 55,
    #              'MABT2.7260.17': 56,
    #              'MABT2.7260.18': 57,
    #              'MABT2.7260.19': 58,
    #              'MABT2.7260.20': 59,
    #              'MABT2.7260.21': 60,
    #              'MABT2.7260.22': 61,
    #              'MABT2.7260.23': 62,
    #              'MABT2.7260.24': 63,
    #              'MABT2.7260.25': 64,
    #              'MABT2.7260.26': 65,
    #              'MABT2.7260.27': 66,
    #              'MABT2.7260.28': 67,
    #              'MABT2.7260.29': 68,
    #              'MABT2.7260.30': 69,
    #              'MABT2.7260.31': 70,
    #              'MABT2.7260.32': 71,
    #              'MABT2.7260.33': 72,
    #              'MABT2.7260.34': 73,
    #              'MABT2.7260.35': 74,
    #              'MABT2.7260.36': 75,
    #              'MABT2.7260.37': 76,
    #              'MABT2.7260.38': 77,
    #              'MABT2.7260.39': 78,
    #              'MABT2.7260.40': 79,
    #              'MABT2.7260.41': 80,
    #              'MABT2.7260.42': 81,
    #              'MABT2.7260.43': 82,
    #              'MABT2.7260.44': 83,
    #              'MABT3.7280.0': 84,
    #              'MABT3.7280.1': 85,
    #              'MABT3.7280.2': 86,
    #              'MABT3.7280.3': 87,
    #              'MABT3.7280.4': 88,
    #              'MABT3.7280.5': 89,
    #              'MABT3.7280.6': 90,
    #              'MABT3.7280.7': 91,
    #              'MABT3.7280.8': 92,
    #              'MABT3.7280.9': 93,
    #              'MABT3.7280.10': 94,
    #              'MABT3.7280.11': 95,
    #              'MABT3.7280.12': 96,
    #              'MABT3.7280.13': 97,
    #              'MABT3.7280.14': 98,
    #              'MABT3.7280.15': 99,
    #              'MABT3.7280.16': 100,
    #              'MABT3.7280.17': 101,
    #              'MABT3.7280.18': 102,
    #              'MABT3.7280.19': 103,
    #              'MABT3.7280.20': 104,
    #              'MABT3.7280.21': 105,
    #              'MABT3.7280.22': 106,
    #              'MABT3.7280.23': 107,
    #              'MABT3.7280.24': 108,
    #              'MABT3.7280.25': 109,
    #              'MABT3.7280.26': 110,
    #              'MABT3.7280.27': 111,
    #              'MABT3.7280.28': 112,
    #              'MABT3.7280.29': 113,
    #              'MABT3.7280.30': 114,
    #              'MABT3.7280.31': 115,
    #              'MABT3.7280.32': 116,
    #              'MABT3.7280.33': 117,
    #              'MABT3.7280.34': 118,
    #              'MABT3.7280.35': 119,
    #              'MABT3.7280.36': 120,
    #              'MABT3.7280.37': 121,
    #              'MABT3.7280.38': 122,
    #              'MABT3.7280.39': 123,
    #              'MABT3.7280.40': 124,
    #              'MABT3.7280.41': 125,
    #              'MABT3.7280.42': 126,
    #              'MABT3.7280.43': 127}
    #
    #     return vocab, len(vocab)
    #
    # def id2label(self):
    #     idx2label = {i: w for i, w in enumerate(self.vocab[0])}
    #     return idx2label
