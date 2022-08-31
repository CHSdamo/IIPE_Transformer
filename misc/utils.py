import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import time
import os

from misc.metrics import metric

from dataset.vmas import VMASDataset, VMASDataset_test, VMASDataset_val, VMASDataset_train
from models.model import Transformer
from torch.optim.lr_scheduler import LambdaLR
from misc.tools import adjust_learning_rate
from sklearn.metrics import f1_score


class Experiment(object):
    def __init__(self, args):
        self.counter = 0
        self.patience = 10
        self.best_score = None
        self.early_stop = False
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = VMASDataset(self.args)
        self.class_weights = self.dataset.class_weights
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model_dict = {
            'transformer': Transformer
        }
        model = model_dict[self.args.model](
            self.dataset,
            # self.args.c_out,
            self.args.d_model,
            self.args.n_heads,
            self.args.enc_layers,
            self.args.dec_layers,
            self.args.d_ff,
            self.args.dropout,
            # self.args.embed,
            # self.args.freq,
            self.args.activation,
            self.args.output_attention,
            self.args.mix,
        ).float()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('The model has {} trainable parameters'.format(total_params))
        return model

    def early_stopping(self, val_loss, model, path, epoch, optimizer):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path, epoch, optimizer)
        elif score > self.best_score + 1e-17:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path, epoch, optimizer)
            self.counter = 0

    def save_checkpoint(self, model, path, epoch, optimizer):
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, path + '/' + 'checkpoint.pth')

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

    def _select_criterion(self, flag, mode=None, weighted=True):
        if flag == 'mse':
            criterion = nn.MSELoss()
        elif flag == 'rmse':
            criterion = self._rmse
        elif flag == 'ce':
            device = torch.device('cuda' if mode == 'train' else 'cpu')
            if weighted:
                weight = torch.tensor(self.class_weights, dtype=torch.float, device=device)
                criterion = nn.CrossEntropyLoss(weight=weight, reduction='mean')
            else:
                criterion = nn.CrossEntropyLoss(reduction='mean')
        return criterion

    def _rmse(self, output, target):
        eps = 1e-8
        mse = nn.MSELoss()
        return torch.sqrt(mse(output, target) + eps)

    def _load_data(self, flag):
        volume = len(self.dataset)
        range1 = [0,                 int(0.7 * volume), int(0.9 * volume)]
        range2 = [int(0.7 * volume), int(0.9 * volume),            volume]
        # range1 = [0, 2000, 4000]
        # range2 = [2000, 4000, 4600]
        if flag == 'train':
            shuffle = True
            drop_last = True
            batch_size = self.args.batch_size
            dataset = VMASDataset_train(self.args, range1[0], range2[0])

        elif flag == 'val':
            shuffle = True
            drop_last = True
            batch_size = self.args.batch_size
            dataset = VMASDataset_val(self.args, range1[1], range2[1])

        elif flag == 'pred':
            shuffle = False
            drop_last = False
            batch_size = 1
            dataset = VMASDataset_test(self.args, range1[2], range2[2])

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)

        return dataset, data_loader

    def train(self, setting):
        train_data, train_loader = self._load_data(flag='train')
        val_data, val_loader = self._load_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        optimizer = self._select_optimizer()
        criterion_rmse = self._select_criterion('rmse')
        criterion_ce_train = self._select_criterion('ce', 'train', weighted=False)
        # scheduler = self._select_scheduler(optimizer, num_warmup_steps=4000, d_model=self.args.d_model)

        train_losses = []
        train_ce_loss = []
        train_rmse_loss = []
        val_losses = []
        lrs = []
        resume = self.args.resume
        for epoch in range(self.args.epochs):
            if resume:
                best_model_path = path + '/' + 'checkpoint.pth'
                # self.model.load_state_dict(torch.load(best_model_path))
                checkpoint = torch.load(best_model_path)
                if len(checkpoint) > 10:
                    self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    epoch = checkpoint['epoch']
                self.model.to(self.device)
                resume = False

            iter_count = 0
            ce_loss = []
            rmse_loss = []
            running_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (sample, pred, tgt, label, src_seq_mark, tgt_seq_mark) in enumerate(tqdm(train_loader)):
                iter_count += 1
                optimizer.zero_grad()

                # dec_logits, reg_output, enc_self_attns, dec_self_attns, dec_enc_attns
                cls_output, reg_output, _ = self.process_one_batch(sample, pred, src_seq_mark, tgt_seq_mark)

                loss1 = criterion_ce_train(cls_output.reshape(-1, cls_output.size(-1)), label.reshape(-1).long())
                loss2 = criterion_rmse(reg_output, pred[:, -self.args.pred_len:, -1].float())
                loss = loss1 + loss2
                running_loss.append(loss.item())
                ce_loss.append(loss1.item())
                rmse_loss.append(loss2.item())

                if (i + 1) % 500 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

                lrs.append(optimizer.param_groups[-1]['lr'])
                loss.backward()
                optimizer.step()
                # scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(running_loss)
            train_losses.append(train_loss)
            train_ce_loss.append(np.average(ce_loss))
            train_rmse_loss.append(np.average(rmse_loss))

            val_loss = self.val(val_loader, criterion_rmse)
            val_losses.append(val_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Val Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, val_loss))
            self.early_stopping(val_loss, self.model, path, epoch, optimizer)
            if self.early_stop:
                print("Early stopping")
                break

        adjust_learning_rate(optimizer, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'

        checkpoint = torch.load(best_model_path)
        if len(checkpoint) > 10:
            self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        # self.model.load_state_dict(checkpoint['model_state_dict'])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'train_loss.npy', np.array(train_losses))
        np.save(folder_path + 'val_loss.npy', np.array(val_losses))
        np.save(folder_path + 'train_ce_loss.npy', np.array(train_ce_loss))
        np.save(folder_path + 'train_rmse_loss.npy', np.array(train_rmse_loss))

        return self.model

    def process_one_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """
        :param batch_x: [batch_size, seq_len, feature(2)]
        :param batch_y: [batch_size, seq_len, feature(2)]
        :param batch_x_mark:  [batch_size, seq_len]
        :param batch_y_mark:  [batch_size, seq_len]
        :return:
        """
        enc_inp = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)  # [bz, pred_len, features]
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        outputs = self.model(enc_inp, dec_inp, batch_x_mark, batch_y_mark)
        # dec_logits, reg_output, enc_self_attns, dec_self_attns, dec_enc_attns = outputs
        dec_logits, reg_output, attns = outputs

        dec_logits = dec_logits[:, -self.args.pred_len:, :]
        reg_output = reg_output[:, -self.args.pred_len:]
        return dec_logits, reg_output, attns  # enc_self_attns, dec_self_attns, dec_enc_attns

    def val(self, val_loader, criterion_rmse):
        self.model.eval()
        total_loss = []
        for i, (sample, pred, tgt, label, src_seq_mark, tgt_seq_mark) in enumerate(tqdm(val_loader)):

            cls_output, reg_output, attns = self.process_one_batch(sample, pred, src_seq_mark, tgt_seq_mark)
            criterion_ce_val = self._select_criterion('ce', 'val', weighted=False)

            loss1 = criterion_ce_val(cls_output.reshape(-1, cls_output.size(-1)).detach().cpu(), label.reshape(-1).long().detach().cpu())
            loss2 = criterion_rmse(reg_output.detach().cpu(), pred[:, -self.args.pred_len:, -1].float().detach().cpu())
            loss = loss1 + loss2
            total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def predict(self, setting):
        pred_data, pred_loader = self._load_data(flag='pred')

        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/' + 'checkpoint.pth'
        checkpoint = torch.load(best_model_path)

        if len(checkpoint) > 10:
            self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()

        gt_labels = []
        gt_tgts = []
        gt_tgts_norm = []
        greedy_labels = []
        greedy_tgts = []
        greedy_tgts_norm = []
        attention_output = []
        f1 = []

        for i, (sample, pred, tgt, label, src_seq_mark, tgt_seq_mark) in enumerate(tqdm(pred_loader)):

            cls_output, reg_output, attns = self.process_one_batch(sample, pred, src_seq_mark, tgt_seq_mark)
            prob = cls_output.squeeze(0).max(dim=-1, keepdim=False)[1]

            reg_output_inv = self.dataset.inverse_data(reg_output, 'Duration')

            greedy_labels.append(prob.detach().cpu().numpy())
            greedy_tgts_norm.append(reg_output.squeeze().detach().cpu().numpy())
            greedy_tgts.append(reg_output_inv.squeeze().detach().cpu().numpy())

            gt_labels.append(label.squeeze().detach().cpu().numpy())
            gt_tgts.append(tgt.squeeze().detach().cpu().numpy())
            gt_tgts_norm.append(pred[:, -self.args.pred_len:, -1].float().squeeze().detach().cpu().numpy())

            attention_output.append(attns[0].squeeze().detach().cpu().numpy())

        save_path = os.path.join('./results/', setting)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path + '/greedy_labels.npy', np.stack(greedy_labels, axis=0))
        np.save(save_path + '/greedy_tgts.npy', np.stack(greedy_tgts, axis=0))
        np.save(save_path + '/greedy_tgts_norm.npy', np.stack(greedy_tgts_norm, axis=0))
        np.save(save_path + '/gt_labels.npy', np.stack(gt_labels, axis=0))
        np.save(save_path + '/gt_tgts.npy', np.stack(gt_tgts, axis=0))
        np.save(save_path + '/gt_tgts_norm.npy', np.stack(gt_tgts_norm, axis=0))
        # np.save(save_path + '/attention.npy', np.stack(attention_output, axis=0))

        greedy_labels = np.load(save_path + '/greedy_labels.npy')
        greedy_tgts_norm = np.load(save_path + '/greedy_tgts_norm.npy')
        greedy_tgts = np.load(save_path + '/greedy_tgts.npy')

        gt_labels = np.load(save_path + '/gt_labels.npy')
        gt_tgts_norm = np.load(save_path + '/gt_tgts_norm.npy')
        gt_tgts = np.load(save_path + '/gt_tgts.npy')

        for i in range(self.args.pred_len):
            f1i = f1_score(gt_labels[:, i], greedy_labels[:, i], average='micro')
            f1.append(f1i)
        np.save(save_path + '/f1.npy', np.asarray(f1))

        mae, mse, rmse, mape, mspe = metric(greedy_tgts_norm, gt_tgts_norm)
        np.save(save_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))

        rmse_i = np.sqrt((np.square(gt_tgts - greedy_tgts)).mean(axis=0))
        np.save(save_path + '/rmse_i.npy', rmse_i)
        rmse_i_norm = np.sqrt((np.square(gt_tgts_norm - greedy_tgts_norm)).mean(axis=0))
        np.save(save_path + '/rmse_i_norm.npy', rmse_i_norm)

        total_correct_acc = np.all(np.equal(gt_labels, greedy_labels), axis=1).sum() / len(gt_labels)
        acc = np.count_nonzero(np.equal(greedy_labels, gt_labels), axis=0) / len(gt_labels)
        np.save(save_path + '/acc.npy', acc)
        np.save(save_path + '/total_correct_acc.npy', total_correct_acc)

        return