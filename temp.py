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
from .network import TripletLoss, FuseNet, CrossEntropyLabelSmooth
from .utils import TripletSampler, TraverseSampler
from .utils.transform import get_blur_transform
from config import conf

import wandb

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

        if self.model_name == 'baseline':
            self.encoder = SetNet(self.hidden_dim)
        else:
            self.encoder = FuseNet(self.hidden_dim)
            print('partnet')
        self.encoder = nn.DataParallel(self.encoder)
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.triplet_loss = nn.DataParallel(self.triplet_loss)
        self.encoder.cuda()
        self.triplet_loss.cuda()
        self.center = torch.zeros(73, 256)
        # self.cross_entropy = nn.CrossEntropyLoss().cuda()
        self.cross_entropy = CrossEntropyLabelSmooth(73).cuda()
        self.center_cross_entropy = CrossEntropyLabelSmooth(73).cuda()
        self.temperature = 0.1
        self._momentum = 0.9

        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
        ], lr=self.lr)

        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.std_metric = []
        self.center_ce_loss_metric = []
        self.ce_loss_metric = []
        self.mean_dist = 0.01
        self.test_acc = 0
        self.best_acc = 0
        self.sample_type = 'random'

    def random_part_erase_mask(self, x, prob=0.5):
        mask = torch.ones_like(x)

        if random.random() > prob:
            return mask
        
        n, s, h, w = x.shape
        assert h==64 and w==44, 'The shape of image should be (64, 44), but got ' + str(x.shape[-2:])
        h_start = random.randint(8, 51)
        mask[:, :, h_start:h_start+4, :] = 0.
        
        return mask

    def collate_fn(self, batch):
        batch_size = len(batch)
        feature_num = len(batch[0][0])
        _seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        # _dt = [batch[i][5] for i in range(batch_size)] 
        batch = [_seqs, view, seq_type, label, None]

        def select_frame(index):
            sample = _seqs[index]
            # dif_sample = _dt[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                if len(frame_set) >= 30:
                    if len(frame_set) > 40:
                        x = random.randint(0, (len(frame_set) - 40))
                        frame_set = frame_set[x:x+40]
                        frame_id_list = np.random.choice(frame_set, size=self.frame_num, replace=False)
                        frame_id_list.sort()
                        _ = [feature.loc[frame_id_list].values for feature in sample]
                        # __ = [feature.loc[frame_id_list].values for feature in dif_sample]
                    else:
                        frame_id_list = np.random.choice(frame_set, size=self.frame_num, replace=False)
                        frame_id_list.sort()
                        _ = [feature.loc[frame_id_list].values for feature in sample]
                        # __ = [feature.loc[frame_id_list].values for feature in dif_sample]
                else:
                    s_frame_id_list = np.random.choice(frame_set, size=(self.frame_num - len(frame_set)), replace=False).tolist()
                    frame_id_list = s_frame_id_list + frame_set
                    frame_id_list.sort()
                    _ = [feature.loc[frame_id_list].values for feature in sample]
                    # __ = [feature.loc[frame_id_list].values for feature in dif_sample]
            else:
                _ = [feature.loc[frame_set].values for feature in sample]
                # __ = [feature.loc[frame_set].values for feature in dif_sample]
            # return _, __
            return _
        seqs = []
        # dt = []
        for seq in map(select_frame, range(len(_seqs))):
            seqs.append(seq)
            # dt.append(dif)

        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
            # dt = [np.asarray([dt[i][j] for i in range(batch_size)]) for j in range(feature_num)]
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


        ############
            # dt = [[
            #                         np.concatenate([
            #                                         dt[i][j]
            #                                         for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
            #                                         if i < batch_size
            #                                         ], 0) for _ in range(gpu_num)]
            #                     for j in range(feature_num)]
            # dt = [np.asarray([
            #                        np.pad(dt[j][_],
            #                               ((0, max_sum_frame - dt[j][_].shape[0]), (0, 0), (0, 0)),
            #                               'constant',
            #                               constant_values=0)
            #                        for _ in range(gpu_num)])
            #         for j in range(feature_num)]
            
            batch[4] = np.asarray(batch_frames)

        # batch.append(dt)
        batch[0] = seqs
        return batch
    
    def init(self, load_model=False, again=False):
        if load_model:
            print('Loading model...', end='')
            ckpt = torch.load('/mnt/md1/huanzhang/iccv/GaitPart/work/checkpoint/GaitPart/GaitPart_CASIA-B_73_False_256_0.2_128_full_30-80000-encoder.ptm')
            model_dict = self.encoder.state_dict()
            for key in model_dict.keys():
                if key in ckpt:
                    model_dict[key] = ckpt[key]
            self.encoder.load_state_dict(model_dict)
            print('OK!')
        else:
            print('laji init')
       
        if os.path.exists('center.pkl') and (not again):
            print('Loading Center Vector...', end='')
            with open('center.pkl', 'rb') as f:
                self.center = pickle.load(f)
            print('OK!')
            self.center = torch.tensor(self.center).cuda()
        else:
            if again:
                print('calculate again')
            print('Calculating Center Vector...', end='')
            traverse_sampler = TraverseSampler(self.train_source, (self.batch_size[0], self.batch_size[1]//8))
            train_loader = tordata.DataLoader(
                dataset=self.train_source,
                batch_sampler=traverse_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers)

            train_label_set = list(self.train_source.label_set)
            train_label_set.sort()
            total_num = torch.zeros(73)
            for seq, view, seq_type, label, batch_frame in train_loader:
                for i in range(len(seq)):
                    seq[i] = self.np2var(seq[i]).float()

                if batch_frame is not None:
                    batch_frame = self.np2var(batch_frame).int()
                feature = self.encoder(*seq, batch_frame)

                target_label = [train_label_set.index(l) for l in label]
                target_labels = self.np2var(np.array(target_label)).long()

                feature = feature.mean(1)+feature.max(1)[0]
                norm_feature = F.normalize(feature, 2, dim=-1)
                
                statistic_mean = norm_feature.clone().detach().cpu()
                for idx, target_label in enumerate(target_labels):
                    self.center[target_label] = self._momentum * self.center[target_label] + (1.-self._momentum)*statistic_mean[idx]
                    total_num[target_label] += 1.
            self.center = self.center / total_num.unsqueeze(-1)
            with open('center.pkl', 'wb') as f:
                pickle.dump(self.center.cpu().numpy(), f)
            print('OK!')
            self.center = self.center.cuda()

    def fit(self):
        print(self.encoder)
        wandb.init(project='memory_dilate_inc')
        wandb.watch_called = False
        config = wandb.config
        config.batch_size = (8, 16)
        config.test_batch_size = 1
        config.log_interval = 100

        wandb.watch(self.encoder, log='all')
        if self.restore_iter == 0:
            self.init(again=True)
        if self.restore_iter != 0:
            self.init(again=True)
        if self.restore_iter != 0:
            self.load_path(self.restore_iter)

        self.encoder.train()
        self.sample_type = 'random'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        self.train_source.set_transform(get_blur_transform(15))
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
                mask = self.random_part_erase_mask(seq[i])
                seq[i] = seq[i] * mask
                # dif[i] = self.np2var(dif[i]).float()
    
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            feature = self.encoder(*seq, batch_frame)

            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()

            triplet_feature = feature.permute(1, 0, 2).contiguous()
            triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)

            feature = feature.mean(1)+feature.max(1)[0] #remove the spatial dimension
            norm_feature = F.normalize(feature, 2, dim=-1) #L2 Normalize
            norm_center = F.normalize(self.center, 2, dim=-1) #L2 Normalize

            # ce_loss = self.cross_entropy(logits.permute(1, 0, 2).reshape(-1, 73), triplet_label.reshape(-1))
            center_logits = torch.matmul(norm_feature, norm_center.transpose(-1, -2)) / self.temperature #[N, 73]
            center_ce_loss = self.center_cross_entropy(center_logits, target_label)

            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
             ) = self.triplet_loss(triplet_feature, triplet_label)
            if self.hard_or_full_trip == 'hard':
                loss = hard_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                loss = full_loss_metric.mean()

            #calculate the id indexs in this batch
            group_feature = norm_feature.reshape(self.batch_size[0], self.batch_size[1], 256)
            statistic_mean = group_feature.mean(1).clone().detach()
            statistic_std = norm_feature.std(1).mean()
            label_idxs = target_label.reshape(self.batch_size[0], self.batch_size[1])[:, 0]

            #update
            self.center[label_idxs] = self._momentum * self.center[label_idxs] + (1.-self._momentum)*statistic_mean

            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())
            # self.ce_loss_metric.append(ce_loss.item())
            self.center_ce_loss_metric.append(center_ce_loss.item())
            self.std_metric.append(statistic_std.item())

            # if self.restore_iter > 40000:
            #     loss = loss + center_ce_loss + statistic_std + 2 * ce_loss
            # else:
            #     loss = loss + ce_loss + statistic_std

            loss = loss + (calWeight(self.restore_iter, self.total_iter)) * center_ce_loss + statistic_std

            if loss > 1e-9:
                loss.backward()
                self.optimizer.step()

            if self.restore_iter % 1000 == 0:
                self.save()
                print(datetime.now() - _time1)
                _time1 = datetime.now()

            if self.restore_iter % 5000 == 0:
                self.test_acc = self.test()
                if self.test_acc > self.best_acc:
                    self.best_acc = self.test_acc
                self.encoder.train()
            
                

            if self.restore_iter % 100 == 0:
                wandb.log({
                    'test accuracy': self.test_acc,
                    'loss': loss,
                    'center_ce_loss':center_ce_loss.item(),
                    'std_metric':np.mean(self.std_metric),
                    'full_loss_metric':np.mean(self.full_loss_metric),
                    'full_loss_num':np.mean(self.full_loss_num),
                    'mean_dist':np.mean(self.dist_list),
                    'center_mean': self.center.mean(),
                    'center_std' : self.center.std(),
                    # 'batch_acc' : batch_center_acc,
                    'cos_weight': calWeight(self.restore_iter, self.total_iter)
                
                })                
                print('iter {}:'.format(self.restore_iter), end='')
                # print(', ce_loss_metric={0:.8f}'.format(np.mean(self.ce_loss_metric)), end='')
                print(', center_ce_loss_metric={0:.8f}'.format(np.mean(self.center_ce_loss_metric)), end='')
                print(', std_metric={0:.8f}'.format(np.mean(self.std_metric)), end='')
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
                self.std_metric = []
                self.ce_loss_metric = []
                self.center_ce_loss_metric = []

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
        self.sample_type = 'all'
        # source = self.train_source
        # self.sample_type = 'random'
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
                # dif[j] = self.np2var(dif[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()
            # print(batch_frame, np.sum(batch_frame))

            feature = self.encoder(*seq, batch_frame)
            n, num_bin, _ = feature.size()
            feature = feature.permute(1, 0, 2).contiguous()
            feature_list.append(feature.data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label

        return np.concatenate(feature_list, 1), view_list, seq_type_list, label_list

    def save(self):
        os.makedirs(osp.join(self.file_path, 'checkpoint', self.model_name), exist_ok=True)
        torch.save(self.encoder.state_dict(),
                   osp.join(self.file_path, 'checkpoint', self.model_name,
                            '{}-{:0>5}-encoder.ptm'.format(
                                self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(),
                   osp.join(self.file_path, 'checkpoint', self.model_name,
                            '{}-{:0>5}-optimizer.ptm'.format(
                                self.save_name, self.restore_iter)))

    # restore_iter: iteration index of the checkpoint to load
    def load(self, restore_iter):
        ckpt = torch.load(osp.join(self.file_path,
            'checkpoint/'+ self.model_name, self.model_name + '_CASIA-B_73_False_256_0.2_128_' + self.hard_or_full_trip + '_30-{:0>5}-encoder.ptm'.format(restore_iter)))
        model_dict = self.encoder.state_dict()
        for key in model_dict.keys():
            if key in ckpt:
                model_dict[key] = ckpt[key]
        self.encoder.load_state_dict(model_dict)
        # self.optimizer.load_state_dict(torch.load(osp.join(self.file_path,
        #     'checkpoint/'+ self.model_name, self.model_name + '_CASIA-B_73_False_256_0.2_128_' + self.hard_or_full_trip + '_30-{:0>5}-optimizer.ptm'.format(restore_iter))))


    def load_path(self, path):
        ckpt = torch.load(path)
        model_dict = self.encoder.state_dict()
        for key in model_dict.keys():
            if key in ckpt:
                model_dict[key] = ckpt[key]
        self.encoder.load_state_dict(model_dict)
        # self.optimizer.load_state_dict(torch.load(osp.join(self.file_path,
        #     'checkpoint/'+ self.model_name, self.model_name + '_CASIA-B_73_False_256_0.2_128_' + self.hard_or_full_trip + '_30-{:0>5}-optimizer.ptm'.format(restore_iter))))

    def test(self):
        print('Transforming...')
        time = datetime.now()
        test = self.transform('test', 1)
        print('Evaluating...')
        acc = evaluation(test, conf['data'])
        print('Evaluation complete. Cost:', datetime.now() - time)

        # Print rank-1 accuracy of the best model
        # e.g.
        # ===Rank-1 (Include identical-view cases)===
        # NM: 95.405,     BG: 88.284,     CL: 72.041
        for i in range(1):
            print('===Rank-%d (Include identical-view cases)===' % (i + 1))
            print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                np.mean(acc[0, :, :, i]),
                np.mean(acc[1, :, :, i]),
                np.mean(acc[2, :, :, i])))

        # Print rank-1 accuracy of the best model，excluding identical-view cases
        # e.g.
        # ===Rank-1 (Exclude identical-view cases)===
        # NM: 94.964,     BG: 87.239,     CL: 70.355
        for i in range(1):
            print('===Rank-%d (Exclude identical-view cases)===' % (i + 1))
            print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                de_diag(acc[0, :, :, i]),
                de_diag(acc[1, :, :, i]),
                de_diag(acc[2, :, :, i])))

        # Print rank-1 accuracy of the best model (Each Angle)
        # e.g.
        # ===Rank-1 of each angle (Exclude identical-view cases)===
        # NM: [90.80 97.90 99.40 96.90 93.60 91.70 95.00 97.80 98.90 96.80 85.80]
        # BG: [83.80 91.20 91.80 88.79 83.30 81.00 84.10 90.00 92.20 94.45 79.00]
        # CL: [61.40 75.40 80.70 77.30 72.10 70.10 71.50 73.50 73.50 68.40 50.00]
        np.set_printoptions(precision=2, floatmode='fixed')
        for i in range(1):
            print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
            print('NM:', de_diag(acc[0, :, :, i], True))
            print('BG:', de_diag(acc[1, :, :, i], True))
            print('CL:', de_diag(acc[2, :, :, i], True))

        return (de_diag(acc[0, :, :, i]) + de_diag(acc[1, :, :, i]) + de_diag(acc[2, :, :, i])) / 3