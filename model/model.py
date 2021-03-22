import math
import os
import os.path as osp
import random
import sys
from datetime import datetime

import numpy as np
import time
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata
import torch.nn.functional as F
import pickle
from .utils import evaluation
from .network import TripletLoss, MTNet
from .utils import TripletSampler
from config import conf
import pdb

import wandb

from .network import TripletLoss, PartNet, MTNet
def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result

def calWeight(i,n):
    ratio=float(i/n)
    x=torch.tensor(ratio*np.pi)
    weight=(torch.sin(x-0.5*np.pi)+1)/2
    return weight.item()

def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result

class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 momentum,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 train_pid_num,
                 frame_num,
                 model_name,
                 train_source,
                 test_source,
                 img_size=64):

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        self.hidden_dim = hidden_dim
        self.momentum = momentum
        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size

        self.restore_iter = restore_iter
        self.total_iter = total_iter

        self.img_size = img_size

        self.encoder = MTNet(self.hidden_dim).float()
        # self.encoder = PartNet(self.hidden_dim).float()
        self.encoder = nn.DataParallel(self.encoder)
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.triplet_loss = nn.DataParallel(self.triplet_loss)
        self.encoder.cuda()
        self.triplet_loss.cuda()

        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
        ], lr=self.lr)

        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01
        self.test_acc = 0
        self.best_acc = 0

        self.sample_type = 'all'

    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                if len(frame_set) >= 30:
                    if len(frame_set) > 40:
                        x = random.randint(0, (len(frame_set) - 40))
                        frame_set = frame_set[x:x+40]
                        frame_id_list = np.random.choice(frame_set, size=self.frame_num, replace=False)
                        frame_id_list.sort()
                        _ = [feature.loc[frame_id_list].values for feature in sample]
                    else:
                        frame_id_list = np.random.choice(frame_set, size=self.frame_num, replace=False)
                        frame_id_list.sort()
                        _ = [feature.loc[frame_id_list].values for feature in sample]
                else:
                    s_frame_id_list = np.random.choice(frame_set, size=(self.frame_num - len(frame_set)), replace=False).tolist()
                    frame_id_list = s_frame_id_list + frame_set
                    frame_id_list.sort()
                    _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                _ = [feature.loc[frame_set].values for feature in sample]
            return _

        seqs = list(map(select_frame, range(len(seqs))))

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                                len(frame_sets[i])
                                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                if i < batch_size
                                ] for _ in range(gpu_num)]
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            seqs = [[
                        np.concatenate([
                                           seqs[i][j]
                                           for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                           if i < batch_size
                                           ], 0) for _ in range(gpu_num)]
                    for j in range(feature_num)]
            seqs = [np.asarray([
                                   np.pad(seqs[j][_],
                                          ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                                          'constant',
                                          constant_values=0)
                                   for _ in range(gpu_num)])
                    for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch

    def fit(self):
        wandb.init(project='MT3D_conv1_nobias')
        wandb.watch_called = False
        config = wandb.config
        config.batch_size = (12, 4)
        config.test_batch_size = 1
        config.log_interval = 100
        wandb.watch(self.encoder, log='all')
        if self.restore_iter != 0:
            self.load(self.restore_iter)


        self.encoder.train()
        self.sample_type = 'random'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        triplet_sampler = TripletSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()

        _time1 = datetime.now()
        for seq, view, seq_type, label, batch_frame in train_loader:
            self.restore_iter += 1
            self.optimizer.zero_grad()

            for i in range(len(seq)):
                seq[i] = self.np2var(seq[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            feature, label_prob = self.encoder(*seq, batch_frame)

            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()

            triplet_feature = feature.permute(1, 0, 2).contiguous()
            # pdb.set_trace()
            triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
             ) = self.triplet_loss(triplet_feature, triplet_label)
            if self.hard_or_full_trip == 'hard':
                loss = hard_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                loss = full_loss_metric.mean()

            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())

            if loss > 1e-9:
                loss.backward()
                self.optimizer.step()

            if self.restore_iter % 1000 == 0:
                self.save()
                print(datetime.now() - _time1)
                _time1 = datetime.now()

            # if self.restore_iter % 5000 == 0:
            #     self.test_acc = self.test()
            #     if self.test_acc > self.best_acc:
            #         self.best_acc = self.test_acc
            #     self.encoder.train()

            if self.restore_iter % 100 == 0:
                wandb.log({
                    # 'test accuracy': self.test_acc,
                    'loss': loss,
                    'hard_loss_metric':np.mean(self.hard_loss_metric),
                    'full_loss_metric':np.mean(self.full_loss_metric),
                    'full_loss_num':np.mean(self.full_loss_num),
                    'mean_dist':np.mean(self.dist_list),
                    # 'batch_acc' : batch_center_acc,
                    'cos_weight': calWeight(self.restore_iter, self.total_iter)
                
                })        
                print('iter {}:'.format(self.restore_iter), end='')
                print(', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                print(', hard or full=%r' % self.hard_or_full_trip)
                sys.stdout.flush()
                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []

            # Visualization using t-SNE
            # if self.restore_iter % 500 == 0:
            #     pca = TSNE(2)
            #     pca_feature = pca.fit_transform(feature.view(feature.size(0), -1).data.cpu().numpy())
            #     for i in range(self.P):
            #         plt.scatter(pca_feature[self.M * i:self.M * (i + 1), 0],
            #                     pca_feature[self.M * i:self.M * (i + 1), 1], label=label[self.M * i])
            #
            #     plt.show()

            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source
        self.sample_type = 'random'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(data_loader):
            seq, view, seq_type, label, batch_frame = x
            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            # print(batch_frame, np.sum(batch_frame))

            feature, _ = self.encoder(*seq, batch_frame)
            # pdb.set_trace()
            n, num_bin, _ = feature.size()
            feature = feature.permute(1, 0, 2).contiguous()
            feature_list.append(feature.data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label

        return np.concatenate(feature_list, 1), view_list, seq_type_list, label_list

    def save(self):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        torch.save(self.encoder.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-encoder.ptm'.format(
                                self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-optimizer.ptm'.format(
                                self.save_name, self.restore_iter)))

    # restore_iter: iteration index of the checkpoint to load
    def load(self, restore_iter):
        self.encoder.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
        self.optimizer.load_state_dict(torch.load(osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))
    
    # def test(self):
    #     print('Transforming...')
    #     time = datetime.now()
    #     test = self.transform('test', 1)
    #     print('Evaluating...')
    #     acc = evaluation(test, conf['data'])
    #     print('Evaluation complete. Cost:', datetime.now() - time)

    #     # Print rank-1 accuracy of the best model
    #     # e.g.
    #     # ===Rank-1 (Include identical-view cases)===
    #     # NM: 95.405,     BG: 88.284,     CL: 72.041
    #     for i in range(1):
    #         print('===Rank-%d (Include identical-view cases)===' % (i + 1))
    #         print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
    #             np.mean(acc[0, :, :, i]),
    #             np.mean(acc[1, :, :, i]),
    #             np.mean(acc[2, :, :, i])))

    #     # Print rank-1 accuracy of the best model，excluding identical-view cases
    #     # e.g.
    #     # ===Rank-1 (Exclude identical-view cases)===
    #     # NM: 94.964,     BG: 87.239,     CL: 70.355
    #     for i in range(1):
    #         print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
    #         print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
    #             de_diag(acc[0, :, :, i]),
    #             de_diag(acc[1, :, :, i]),
    #             de_diag(acc[2, :, :, i])))

    #     # Print rank-1 accuracy of the best model (Each Angle)
    #     # e.g.
    #     # ===Rank-1 of each angle (Exclude identical-view cases)===
    #     # NM: [90.80 97.90 99.40 96.90 93.60 91.70 95.00 97.80 98.90 96.80 85.80]
    #     # BG: [83.80 91.20 91.80 88.79 83.30 81.00 84.10 90.00 92.20 94.45 79.00]
    #     # CL: [61.40 75.40 80.70 77.30 72.10 70.10 71.50 73.50 73.50 68.40 50.00]
    #     np.set_printoptions(precision=2, floatmode='fixed')
    #     for i in range(1):
    #         print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
    #         print('NM:', de_diag(acc[0, :, :, i], True))
    #         print('BG:', de_diag(acc[1, :, :, i], True))
    #         print('CL:', de_diag(acc[2, :, :, i], True))

    #     return (de_diag(acc[0, :, :, i]) + de_diag(acc[1, :, :, i]) + de_diag(acc[2, :, :, i])) / 3
