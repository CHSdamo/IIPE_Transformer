import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import time
import os

from dataset.vmas import VMASDataset
from models.model import Transformer
from torch.optim.lr_scheduler import LambdaLR


class Experiment(object):
    def __init__(self, args):
        self.counter = None
        self.patience = 3
        self.best_score = None
        self.early_stop = False
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = VMASDataset(self.args)
        self.split_dataset()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model_dict = {
            'transformer': Transformer,
        }
        model = model_dict[self.args.model](self.args, self.dataset)

        return model

    def early_stopping(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + 1e-17:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')

    def _select_scheduler(self, optimizer, num_warmup_steps, d_model, last_epoch=-1):
        def lr_lambda(current_step):
            current_step += 1
            arg1 = current_step ** -0.5
            arg2 = current_step * (num_warmup_steps ** -1.5)
            return (d_model ** -0.5) * min(arg1, arg2)

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    def _select_optimizer(self):
        if self.args.optim == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optim == 'sdg':
            optimizer = optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.99)
        # optimizer = NoamOpt(self.args.d_model, 2, 4000,
        #                     torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        return optimizer

    def _select_criterion(self, flag):
        if flag == 'mse':
            criterion = nn.MSELoss()
        elif flag == 'rmse':
            criterion = self._rmse
        elif flag == 'ce':
            criterion = nn.CrossEntropyLoss()
        return criterion

    def _rmse(self, output, target):
        eps = 1e-8
        mse = nn.MSELoss()
        return torch.sqrt(mse(output, target) + eps)

    def _load_data(self, flag):
        if flag == 'test':
            shuffle = False
            drop_last = True
            batch_size = self.args.batch_size
            dataset = self.test_dataset

        elif flag == 'pred':
            shuffle = False
            drop_last = False
            batch_size = 1
            dataset = self.test_dataset

        elif flag == 'train':
            shuffle = True
            drop_last = True
            batch_size = self.args.batch_size
            dataset = self.train_dataset

        elif flag == 'val':
            shuffle = True
            drop_last = True
            batch_size = self.args.batch_size
            dataset = self.val_dataset

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last, )

        return dataset, data_loader

    def split_dataset(self):
        total = len(self.dataset)
        train_num = int(0.7 * total)
        # train_num = int(0.6 * total)
        val_num = int(0.2 * total)
        # val_test_num = total - train_num
        # val_num = int(val_test_num * 0.5)
        test_num = total - val_num - train_num

        assert total == train_num + val_num + test_num

        lengths = [train_num, val_num, test_num]
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.dataset.random_split(self.dataset,
                                                                                                        lengths)

    def train(self, setting):
        train_data, train_loader = self._load_data(flag='train')
        val_data, val_loader = self._load_data(flag='val')
        # test_data, test_loader = self._load_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        optimizer = self._select_optimizer()
        criterion_rmse = self._select_criterion('rmse')
        criterion_ce = self._select_criterion('ce')
        scheduler = self._select_scheduler(optimizer, num_warmup_steps=4000, d_model=self.args.d_model)

        train_losses = []
        ce_losses = []
        rmse_losses = []
        val_losses = []

        for epoch in range(self.args.epochs):
            iter_count = 0
            ce_loss = []
            rmse_loss = []
            running_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (sample, pred, tgt, src_seq_mark, tgt_seq_mark) in enumerate(tqdm(train_loader)):
                iter_count += 1
                optimizer.zero_grad()

                # dec_logits, reg_output, enc_self_attns, dec_self_attns, dec_enc_attns
                cls_output, reg_output, _, _, _ = self.process_one_batch(sample, pred, tgt, src_seq_mark, tgt_seq_mark)

                loss1 = criterion_ce(cls_output, pred[:, :, -2].view(-1).long())
                loss2 = criterion_rmse(reg_output, tgt.float())
                loss = loss1 + loss2
                running_loss.append(loss.item())
                ce_loss.append(loss1.item())
                rmse_loss.append(loss2.item())

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

                loss.backward()
                optimizer.step()
                scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(running_loss)
            train_losses.append(train_loss)
            ce_losses.append(np.average(ce_loss))
            rmse_losses.append(np.average(rmse_loss))

            val_loss = self.val(val_data, val_loader, criterion_ce, criterion_rmse)
            val_losses.append(val_loss)
            # test_loss = self.val(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, val_loss))
            self.early_stopping(val_loss, self.model, path)
            if self.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'train_loss.npy', np.array(train_losses))
        np.save(folder_path + 'val_loss.npy', np.array(val_losses))
        np.save(folder_path + 'ce_loss.npy', np.array(ce_losses))
        np.save(folder_path + 'rmse_loss.npy', np.array(rmse_losses))

        return self.model

    def process_one_batch(self, batch_x, batch_y, tgt, batch_x_mark, batch_y_mark):
        # print(batch_x.size(), batch_y.size())
        batch_x = batch_x.float().to(self.device)
        dec_inp = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        # dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        # dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        dec_logits, reg_output, enc_self_attns, dec_self_attns, dec_enc_attns \
            = self.model(batch_x, dec_inp, batch_x_mark, batch_y_mark)

        reg_output = self.dataset.decode_data(reg_output, 'Duration')

        return dec_logits, reg_output, enc_self_attns, dec_self_attns, dec_enc_attns

    def val(self, val_data, val_loader, criterion_ce, criterion_rmse):
        self.model.eval()
        total_loss = []
        for i, (sample, pred, tgt, src_seq_mark, tgt_seq_mark) in enumerate(tqdm(val_loader)):
            # cls_output, reg_output, err_type_logits, _, _, _ = self.process_one_batch(sample, pred)
            cls_output, reg_output, _, _, _ = self.process_one_batch(sample, pred, tgt, src_seq_mark, tgt_seq_mark)
            loss1 = criterion_ce(cls_output.detach().cpu(), pred[:, :, -2].view(-1).long().detach().cpu())
            loss2 = criterion_rmse(reg_output.detach().cpu(), tgt.float().detach().cpu())
            loss = loss1 + loss2
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting):
        test_data, test_loader = self._load_data(flag='test')
        self.model.eval()

        cls_outputs = []
        reg_outputs = []
        for i, (sample, pred) in enumerate(tqdm(test_loader)):
            cls_output, reg_output, _, _, _ = self.process_one_batch(sample, pred)
            cls_outputs.append(cls_output.detach().cpu().numpy())
            reg_outputs.append(reg_output.detach().cpu().numpy())

        cls_outputs = np.array(cls_outputs)
        reg_outputs = np.array(reg_outputs)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
