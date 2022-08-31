import os
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from misc.tools import StandardScaler
import pandas as pd
import numpy as np
import torch
import torch.nn
from misc.tools import Vocabulary
from sklearn.utils.class_weight import compute_class_weight


class VMASDataset(Dataset):
    def __init__(self, args):

        self.class_weights = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder_fit = {}
        self.data = None
        self.sample = []
        self.pred = []
        self.labels = []
        self.targets = []
        self.src_seq_marks = []
        self.tgt_seq_marks = []
        self.window_label = []
        self.args = args
        # self.raw_df = self.read_csv_file()
        self.raw_df = self.read_tsv_file()
        self.processed_df = self.processed_tsv()
        # self.processed_df = self.preprocess()
        self.dataset_len = len(self.processed_df)
        self.encode_data()
        self.trans_samples()
        # self.prepare_samples()   # for data viewer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = torch.tensor(self.data[index]).to(self.device)
        pred = torch.tensor(self.pred[index]).to(self.device)
        target = torch.tensor(self.targets[index]).to(self.device)
        label = torch.tensor(self.labels[index]).to(self.device)
        src_seq_mark = torch.tensor(self.src_seq_marks[index]).to(self.device)
        tgt_seq_mark = torch.tensor(self.tgt_seq_marks[index]).to(self.device)

        return sample, pred, target, label, src_seq_mark, tgt_seq_mark

    def prepare_samples(self):
        trans_data = np.load('./saves/multistation/trans_data.npy')

        for i in range(0, self.dataset_len - self.args.seq_len - self.args.pred_len, self.args.stride):  # (0, 431990, 1)
            sample = trans_data[i:i + self.args.seq_len][:, 1]
            pred = trans_data[i + self.args.seq_len - self.args.label_len: i + self.args.seq_len + self.args.pred_len][
                   :, 1]

            label = trans_data[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len][:, -2]
            target = trans_data[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len][:, 1]

            src_seq_mark = trans_data[i:i + self.args.seq_len][:, 2]
            tgt_seq_mark = trans_data[
                           i + self.args.seq_len - self.args.label_len: i + self.args.seq_len + self.args.pred_len][:,
                           2]

            self.sample.append(sample)
            self.pred.append(pred)
            self.labels.append(label)
            self.targets.append(target)
            self.src_seq_marks.append(src_seq_mark)
            self.tgt_seq_marks.append(tgt_seq_mark)

        # os.makedirs('./saves/testing', exist_ok=True)
        # np.save('./saves/testing/sample.npy', np.asarray(self.sample))
        # np.save('./saves/testing/pred.npy', np.asarray(self.pred))
        assert len(self.pred) == len(self.labels) == len(self.targets)

    def trans_samples(self):
        if not os.path.exists('./saves/multistation/trans_data.npy'):
            trans_data, trans_label, trans_target = [], [], []

            for _, row in self.processed_df.iterrows():
                row = list(row)
                row[-2] = Vocabulary().label2id()[row[-2]]
                trans_data.append(row)
            trans_data = np.array(trans_data)
            np.save('./saves/multistation/trans_data.npy', trans_data)

        self.class_weights = [0.7793, 1.1245, 1.1247, 1.1247, 1.1245, 1.1160, 1.1247, 0.7114,
                              1.1159, 1.1147, 1.1246, 1.1246, 1.1245, 1.1247, 0.5198, 1.1245,
                              1.1247, -1e9, -1e9, -1e9, -1e9, -1e9, 1.0904, -1e9, -1e9]

        self.data = np.load('./saves/multistation/trans_data.npy')

    def encode_data(self):
        label_colums = ['Car Code']  # , 'Error Types']
        for col_name in label_colums:
            col_fit, col_data = self.encoder_fit_transform(self.processed_df[col_name])
            self.encoder_fit[col_name] = col_fit
            self.processed_df[col_name + '@encoded'] = col_data

        col_fit, col_data = self.encoder_fit_transform(self.processed_df[['Duration']], enc_type='value')
        self.encoder_fit['Duration'] = col_fit
        self.processed_df['Duration' + '@encoded'] = col_data
        sub_col = ['Car Code', 'Duration', 'Car Code@encoded', 'Label', 'Duration@encoded']
        self.processed_df = self.processed_df[sub_col]
        self.processed_df.to_csv('./saves/multistation/encoded.csv')

    def inverse_data(self, data, tag):
        return self.encoder_fit[tag].inverse_transform(data)

    def label_class(self):
        return len(self.encoder_fit['Car Code'].classes_)

    @staticmethod
    def encoder_fit_transform(column, enc_type="label"):
        if enc_type == "label":
            enc = LabelEncoder()
        else:
            enc = StandardScaler()
        enc.fit(column.values)
        return enc, enc.transform(column.values)

    def read_tsv_file(self):
        file = './saves/multistation/raw_df.csv'
        if not os.path.exists(file):
            ev_df = pd.read_csv(self.args.folder + 'Export_LEGATO_20210111_1021_20210517_0559.evzip.tsv', sep='\t')
            ev_df.rename(columns={'Event': 'Action', 'x1': 'Duration', 'Code': 'Car Code'}, inplace=True)
            df_ac = ev_df[ev_df['Action'].str.contains('AC')]
            df_ac['Duration'] = df_ac['Duration'].apply(lambda x: float(x.replace(',', '.'))).astype(float)
            df_ac['Label'] = df_ac[['Section', 'Station', 'Action']].astype(str).agg('.'.join, axis=1)
            df_ac['Car Code'] = pd.to_numeric(df_ac['Car Code'], errors='coerce').astype('Int64')
            df_ac = df_ac[df_ac['Station'].isin([7220, 7230, 7240])]
            os.makedirs('./saves/multistation', exist_ok=True)
            df_ac.to_csv(file, index=False)
        elif not os.path.exists('./saves/multistation/processed_df.csv'):
            df_ac = pd.read_csv(file)
        else:
            df_ac = None
        return df_ac     # (648964, )

    def processed_tsv(self):
        file = './saves/multistation/processed_df.csv'
        if not os.path.exists(file):
            processed_df = self.raw_df
            outlier_df = self.raw_df[self.raw_df['Duration'] >= 600]
            #tmp_df = processed_df[['Area','Section','Station','Timestamp','Action','UniqueID','Car Code', 'Duration']]
            #need_df = tmp_df[tmp_df['UniqueID'] == 12202051410288]
            #need_df['Car Code'] = need_df['Car Code'].astype(int)
            for _, row in outlier_df.iterrows():
                condition = (self.raw_df['Car Code'] == row['Car Code']) & (self.raw_df['Station'] == row['Station']) & (self.raw_df['UniqueID'] == row['UniqueID'])
                processed_df.drop(processed_df[condition].index, inplace=True)
            processed_df.reset_index(inplace=True)

            # for id in self.raw_df[self.raw_df['Duration'] >= 600]['UniqueID'].unique():
            #     self.raw_df[self.raw_df['UniqueID'] == id]
            #     id = 1
            os.makedirs('./saves/multistation', exist_ok=True)
            processed_df.to_csv(file, index=False)
        else:
            processed_df = pd.read_csv('./saves/multistation/processed_df.csv')
            # processed_df.drop(np.concatenate([np.arange(304697, 304699), np.arange(1425577, 1425595),
            #                                   np.arange(1127658, 1127682), np.arange(304718, 304739),
            #                                   np.arange(524619, 524636), np.arange(1297943, 1297950)]), inplace=True)
        return processed_df

    def read_csv_file(self):
        file = './saves/onestation/merged_df.csv'
        if not os.path.exists(file):
            error_type_df = pd.read_csv(self.args.folder + '7carCodes_ACT_All.csv')
            # read tsv file, seperate with Space
            ev_zip_df = pd.read_csv(self.args.folder + 'Export_LEGATO_20210111_1021_20210517_0559.evzip.tsv', sep='\t')
            error_type_df.rename(
                columns={'Event': 'Action', 'Fehlertyp': 'Error Types', 'Fehler Anzahl': 'Error Counts',
                         'Time_diff(s)': 'Duration', 'Code': 'Car Code'},
                inplace=True)
            ev_zip_without_ev_df = ev_zip_df[ev_zip_df['Event'].str.contains('AC')]  # del EV no ev

            join = pd.merge(error_type_df, ev_zip_without_ev_df, on='Timestamp', how='left')
            # del duplicate:
            join = join[join['Duration'] == join['x1'].apply(lambda x: float(x.replace(',', '.'))).astype(float)]
            merged_df = join[
                join['Action'] == join['Event'].str.replace('AC', '').astype(int)]  # del dup in action & event

            merged_df['Label'] = merged_df[['Section', 'Station', 'Action']].astype(str).agg('.'.join, axis=1)
            os.makedirs('./saves/onestation', exist_ok=True)
            merged_df.to_csv(file, index=False)
        else:
            merged_df = pd.read_csv(file)
        return merged_df  # (648964, )

    def preprocess(self):
        file = './saves/onestation/processed_df.csv'
        if not os.path.exists(file):
            # delete whole seq with Time_diff>600 component:
            df = self.raw_df[
                ~self.raw_df['UniqueID'].isin(
                    self.raw_df[self.raw_df['Duration'] >= 600]['UniqueID'].unique())]  # (617159)
            df.reset_index(inplace=True)
            # df.drop(['Error Counts', 'Event', 'Unnamed: 6', 'Unnamed: 7', 'Code_y', 'x1', 'index', 'Section',
            #          'Station', 'Action', 'Timestamp', 'Area', 'UniqueID', 'Unnamed: 0'], axis=1, inplace=True)
            # df['UniqueID'] = np.floor(pd.to_numeric(join['UniqueID'], errors='coerce')).astype('Int64')
            sub_col = ['Car Code', 'Label', 'Duration']  # , 'Error Types']
            processed_df = df[sub_col]
            os.makedirs('./saves/onestation', exist_ok=True)
            processed_df.to_csv(file, index=False)
        else:
            processed_df = pd.read_csv(file)
        # car_code = 1111110022
        # one_car_df = processed_df[processed_df['Car Code'] == car_code]
        return processed_df


class VMASDataset_train(Dataset):
    def __init__(self, args, range1, range2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tgt_seq_marks = []
        self.src_seq_marks = []
        self.targets = []
        self.labels = []
        self.pred = []
        self.data = []
        self.args = args
        self.range1 = range1
        self.range2 = range2
        self.prepare_samples()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = torch.tensor(self.data[index]).to(self.device)
        pred = torch.tensor(self.pred[index]).to(self.device)
        target = torch.tensor(self.targets[index]).to(self.device)
        label = torch.tensor(self.labels[index]).to(self.device)
        src_seq_mark = torch.tensor(self.src_seq_marks[index]).to(self.device)
        tgt_seq_mark = torch.tensor(self.tgt_seq_marks[index]).to(self.device)

        return sample, pred, target, label, src_seq_mark, tgt_seq_mark

    def prepare_samples(self):
        trans_data = np.load('./saves/multistation/trans_data.npy')

        for i in range(self.range1, self.range2 - self.args.seq_len - self.args.pred_len,
                       self.args.stride):  # (0, 431990, 1)
            sample = trans_data[i:i + self.args.seq_len][:, 3:]
            pred = trans_data[i + self.args.seq_len - self.args.label_len: i + self.args.seq_len + self.args.pred_len][
                   :, 3:]

            label = trans_data[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len][:, -2]
            target = trans_data[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len][:, 1]

            src_seq_mark = trans_data[i:i + self.args.seq_len][:, 2]
            tgt_seq_mark = trans_data[
                           i + self.args.seq_len - self.args.label_len: i + self.args.seq_len + self.args.pred_len][:,
                           2]

            self.data.append(sample)
            self.pred.append(pred)
            self.labels.append(label)
            self.targets.append(target)
            self.src_seq_marks.append(src_seq_mark)
            self.tgt_seq_marks.append(tgt_seq_mark)

        assert len(self.pred) == len(self.labels) == len(self.targets)


class VMASDataset_val(Dataset):
    def __init__(self, args, range1, range2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tgt_seq_marks = []
        self.src_seq_marks = []
        self.targets = []
        self.labels = []
        self.pred = []
        self.data = []
        self.args = args
        self.range1 = range1
        self.range2 = range2
        self.prepare_samples()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = torch.tensor(self.data[index]).to(self.device)
        pred = torch.tensor(self.pred[index]).to(self.device)
        target = torch.tensor(self.targets[index]).to(self.device)
        label = torch.tensor(self.labels[index]).to(self.device)
        src_seq_mark = torch.tensor(self.src_seq_marks[index]).to(self.device)
        tgt_seq_mark = torch.tensor(self.tgt_seq_marks[index]).to(self.device)

        return sample, pred, target, label, src_seq_mark, tgt_seq_mark

    def prepare_samples(self):
        trans_data = np.load('./saves/multistation/trans_data.npy')

        for i in range(self.range1, self.range2 - self.args.seq_len - self.args.pred_len,
                       self.args.stride):  # (0, 617201, 1)
            sample = trans_data[i:i + self.args.seq_len][:, 3:]
            pred = trans_data[i + self.args.seq_len - self.args.label_len: i + self.args.seq_len + self.args.pred_len][
                   :, 3:]

            label = trans_data[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len][:, -2]
            target = trans_data[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len][:, 1]

            src_seq_mark = trans_data[i:i + self.args.seq_len][:, 2]
            tgt_seq_mark = trans_data[
                           i + self.args.seq_len - self.args.label_len: i + self.args.seq_len + self.args.pred_len][:,
                           2]

            self.data.append(sample)
            self.pred.append(pred)
            self.labels.append(label)
            self.targets.append(target)
            self.src_seq_marks.append(src_seq_mark)
            self.tgt_seq_marks.append(tgt_seq_mark)

        assert len(self.pred) == len(self.labels) == len(self.targets)


class VMASDataset_test(Dataset):
    def __init__(self, args, range1, range2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tgt_seq_marks = []
        self.src_seq_marks = []
        self.targets = []
        self.labels = []
        self.pred = []
        self.data = []
        self.args = args
        self.range1 = range1
        self.range2 = range2
        self.prepare_samples()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = torch.tensor(self.data[index]).to(self.device)
        pred = torch.tensor(self.pred[index]).to(self.device)
        target = torch.tensor(self.targets[index]).to(self.device)
        label = torch.tensor(self.labels[index]).to(self.device)
        src_seq_mark = torch.tensor(self.src_seq_marks[index]).to(self.device)
        tgt_seq_mark = torch.tensor(self.tgt_seq_marks[index]).to(self.device)

        return sample, pred, target, label, src_seq_mark, tgt_seq_mark

    def prepare_samples(self):
        trans_data = np.load('./saves/multistation/trans_data.npy')

        for i in range(self.range1, self.range2 - self.args.seq_len - self.args.pred_len,
                       self.args.stride):  # (0, 617201, 1)
            sample = trans_data[i:i + self.args.seq_len][:, 3:]
            pred = trans_data[i + self.args.seq_len - self.args.label_len: i + self.args.seq_len + self.args.pred_len][
                   :, 3:]

            label = trans_data[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len][:, -2]
            target = trans_data[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len][:, 1]

            src_seq_mark = trans_data[i:i + self.args.seq_len][:, 2]
            tgt_seq_mark = trans_data[
                           i + self.args.seq_len - self.args.label_len: i + self.args.seq_len + self.args.pred_len][:,
                           2]

            self.data.append(sample)
            self.pred.append(pred)
            self.labels.append(label)
            self.targets.append(target)
            self.src_seq_marks.append(src_seq_mark)
            self.tgt_seq_marks.append(tgt_seq_mark)

        assert len(self.pred) == len(self.labels) == len(self.targets)
