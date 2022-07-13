import os

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import pandas as pd
import logging
import tqdm
import numpy as np
import torch
from dataset.vocab import Vocabulary


class VMASDataset(Dataset):
    def __init__(self, args):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoded_df = None
        self.encoder_fit = {}
        self.dataset_len = None
        self.data = []
        self.pred = []
        self.labels = []
        self.targets = []
        self.window_label = []
        self.args = args
        self.raw_df = self.read_csv_file()
        self.processed_df = self.preprocess()
        self.encode_data()
        self.prepare_samples()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = torch.tensor(self.data[index]).to(self.device)
        pred = torch.tensor(self.pred[index]).to(self.device)

        return sample, pred


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
        col_data = self.encoder_fit_transform(self.processed_df[['Car Code']])
        self.processed_df['Car Code'] = col_data
        col_data = self.encoder_fit_transform(self.processed_df[['Duration']], enc_type='value')
        self.processed_df['Duration'] = col_data

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
