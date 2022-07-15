import os

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder  # , StandardScaler, MinMaxScaler
from misc.tools import StandardScaler
import pandas as pd
import logging
import tqdm
import numpy as np
import torch
# from dataset.vocab import Vocabulary
from misc.tools import Vocabulary


class VMASDataset(Dataset):
    def __init__(self, args):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.encoded_df = None
        self.encoder_fit = {}
        self.data = []
        self.pred = []
        self.labels = []
        self.targets = []
        self.src_seq_marks = []
        self.tgt_seq_marks = []
        self.window_label = []
        self.args = args
        self.raw_df = self.read_csv_file()
        self.processed_df = self.preprocess()
        self.dataset_len = len(self.processed_df)
        self.encode_data()
        self.prepare_samples()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = torch.tensor(self.data[index]).to(self.device)
        pred = torch.tensor(self.pred[index]).to(self.device)
        target = torch.tensor(self.targets[index]).to(self.device)
        src_seq_mark = torch.tensor(self.src_seq_marks[index]).to(self.device)
        tgt_seq_mark = torch.tensor(self.tgt_seq_marks[index]).to(self.device)

        return sample, pred, target, src_seq_mark, tgt_seq_mark


    @staticmethod
    def encoder_fit_transform(column, enc_type="label"):
        if enc_type == "label":
            enc = LabelEncoder()
        else:
            enc = StandardScaler()
        enc.fit(column.values)
        # if inverse:
        #     return enc.inverse_transform(column)
        # else:
        return enc, enc.transform(column.values)

    def prepare_samples(self):
        if not os.path.exists('./saves/trans_data.npy'):
            trans_data, trans_label, trans_target = [], [], []

            for _, row in self.processed_df.iterrows():
                row = list(row)

                row[-2] = Vocabulary().label2id()[row[-2]]
                # trans_data.append(row[2:])
                trans_data.append(row)
                # trans_label.append(row[-2])
                # trans_target.append(row[1])
            trans_data = np.array(trans_data)
            np.save('./saves/trans_data.npy', trans_data)
            # np.save('./saves/trans_label.npy', np.array(trans_label))
            # np.save('./saves/trans_target.npy', np.array(trans_target))
        else:
            # trans_data = np.load('./saves/trans_data.npy').tolist()
            # trans_label = np.load('./saves/trans_label.npy').tolist()
            # trans_target = np.load('./saves/trans_target.npy').tolist()
            trans_data = np.load('./saves/trans_data.npy')
            # trans_label = np.load('./saves/trans_label.npy')
            # trans_target = np.load('./saves/trans_target.npy')

        for i in range(0, len(trans_data) - self.args.seq_len - self.args.pred_len, self.args.stride):  # (0, 617201, 1)
            sample = trans_data[i:i + self.args.seq_len][:, 3:]
            pred = trans_data[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len][:, 3:]
            # label = trans_label[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len]
            # target = trans_target[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len]
            label = trans_data[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len][:, -2]
            target = trans_data[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len][:, 1]
            src_seq_mark = trans_data[i:i + self.args.seq_len][:, 2]
            tgt_seq_mark = trans_data[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len][:, 2]

            self.data.append(sample)
            self.pred.append(pred)
            self.labels.append(label)
            self.targets.append(target)
            self.src_seq_marks.append(src_seq_mark)
            self.tgt_seq_marks.append(tgt_seq_mark)

        assert len(self.pred) == len(self.labels) == len(self.targets)

    def encode_data(self):
        # self.encoded_df = self.processed_df.copy()
        label_colums = ['Car Code']   # , 'Error Types']
        for col_name in label_colums:
            col_fit, col_data = self.encoder_fit_transform(self.processed_df[col_name])
            self.encoder_fit[col_name] = col_fit
            self.processed_df[col_name+'@encoded'] = col_data

        col_fit, col_data = self.encoder_fit_transform(self.processed_df[['Duration']], enc_type='value')
        self.encoder_fit['Duration'] = col_fit
        self.processed_df['Duration'+'@encoded'] = col_data
        sub_col = ['Car Code', 'Duration', 'Car Code@encoded', 'Label', 'Duration@encoded']
        self.processed_df = self.processed_df[sub_col]

    def decode_data(self, data, tag):
        return self.encoder_fit[tag].inverse_transform(data)

    def label_class(self):
        return len(self.encoder_fit['Car Code'].classes_)

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
            sub_col = ['Car Code', 'Label', 'Duration']  # 'Error Types'
            processed_df = df[sub_col]
            os.makedirs('./saves', exist_ok=True)
            processed_df.to_csv(file, index=False)
        else:
            processed_df = pd.read_csv(file)
        return processed_df
