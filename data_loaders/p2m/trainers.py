import os

import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
import torch.optim as optim
from collections import OrderedDict
import time
from os.path import join as pjoin
import torch.nn.functional as F
import codecs as cs
import math
from torch.utils.tensorboard import SummaryWriter

def print_current_loss_decomp(start_time, niter_state, total_niters, losses, epoch=None, inner_iter=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    print('epoch: %03d inner_iter: %5d' % (epoch, inner_iter), end=" ")
    # now = time.time()
    message = '%s niter: %07d completed: %3d%%)'%(time_since(start_time, niter_state / total_niters), niter_state, niter_state / total_niters * 100)
    for k, v in losses.items():
        message += ' %s: %.4f ' % (k, v)
    print(message)


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class PoseMotionMatchTrainer(object):
    def __init__(self, args, pose_encoder, motion_encoder):
        self.opt = args
        self.pose_encoder = pose_encoder
        self.motion_encoder = motion_encoder
        self.device = args.device

        # Training
        writer = SummaryWriter(self.opt.model_dir)
        self.writer = writer
        self.contrastive_loss = ContrastiveLoss(self.opt.negative_margin)


        os.makedirs(self.opt.model_dir, exist_ok=True)

    def save(self, model_dir, epoch, niter):
        state = {
            'pose_encoder': self.pose_encoder.state_dict(),
            'motion_encoder': self.motion_encoder.state_dict(),

            'opt_pose_encoder': self.opt_pose_encoder.state_dict(),
            'opt_motion_encoder': self.opt_motion_encoder.state_dict(),
            'epoch': epoch,
            'iter': niter,
        }
        torch.save(state, model_dir)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def to(self, device):
        self.pose_encoder.to(device)
        self.motion_encoder.to(device)

    def forward(self, batch_data):
        motion, cond = batch_data

        motion = motion.to(self.device)
        cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}

        '''Movement Encoding'''
        self.motion_embedding = self.motion_encoder(motion.squeeze(-2).permute(0, 2, 1))

        '''Text Encoding'''
        self.pose_embedding = self.pose_encoder(cond['y']['pose_feature'])

    def backward(self):

        batch_size = self.pose_embedding.shape[0]
        '''Positive pairs'''
        pos_labels = torch.zeros(batch_size).to(self.pose_embedding.device)
        self.loss_pos = self.contrastive_loss(self.pose_embedding, self.motion_embedding, pos_labels)

        '''Negative Pairs, shifting index'''
        neg_labels = torch.ones(batch_size).to(self.pose_embedding.device)
        shift = np.random.randint(0, batch_size-1)
        new_idx = np.arange(shift, batch_size + shift) % batch_size
        self.mis_motion_embedding = self.motion_embedding.clone()[new_idx]
        self.loss_neg = self.contrastive_loss(self.pose_embedding, self.mis_motion_embedding, neg_labels)
        self.loss = self.loss_pos + self.loss_neg

        loss_logs = OrderedDict({})
        loss_logs['loss'] = self.loss.item()
        loss_logs['loss_pos'] = self.loss_pos.item()
        loss_logs['loss_neg'] = self.loss_neg.item()
        return loss_logs

    def update(self):

        self.zero_grad([self.opt_motion_encoder, self.opt_pose_encoder])
        loss_logs = self.backward()
        self.loss.backward()
        self.clip_norm([self.pose_encoder, self.motion_encoder])
        self.step([self.opt_pose_encoder, self.opt_motion_encoder])

        return loss_logs

    def train(self, train_dataloader, val_dataloader):
        self.to(self.device)

        self.opt_motion_encoder = optim.Adam(self.motion_encoder.parameters(), lr=self.opt.lr)
        self.opt_pose_encoder = optim.Adam(self.pose_encoder.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        logs = OrderedDict()

        min_val_loss = np.inf
        while epoch < self.opt.max_epoch:
            # time0 = time.time()
            for i, batch_data in enumerate(train_dataloader):
                self.pose_encoder.train()
                self.motion_encoder.train()

                self.forward(batch_data)
                log_dict = self.update()
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    self.writer.add_scalar('val_loss', val_loss, it)

                    for tag, value in logs.items():
                        self.writer.add_scalar(tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss_decomp(start_time, it, total_iters, mean_loss, epoch, i)

                    # if it % self.opt.save_latest == 0:
                    #     self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, it)

            print('Validation time:')

            loss_pos_pair = 0
            loss_neg_pair = 0
            val_loss = 0
            with torch.no_grad():
                self.pose_encoder.eval()
                self.motion_encoder.eval()
                for i, batch_data in enumerate(val_dataloader):
                    self.forward(batch_data)
                    self.backward()
                    loss_pos_pair += self.loss_pos.item()
                    loss_neg_pair += self.loss_neg.item()
                    val_loss += self.loss.item()

            loss_pos_pair /= len(val_dataloader) + 1
            loss_neg_pair /= len(val_dataloader) + 1
            val_loss /= len(val_dataloader) + 1
            print('Validation Loss: %.5f Positive Loss: %.5f Negative Loss: %.5f' %
                  (val_loss, loss_pos_pair, loss_neg_pair))

            if val_loss < min_val_loss:
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                min_val_loss = val_loss

            if epoch % self.opt.eval_every_e == 0:
                pos_dist = F.pairwise_distance(self.pose_embedding, self.motion_embedding)
                neg_dist = F.pairwise_distance(self.pose_embedding, self.mis_motion_embedding)

                pos_str = ' '.join(['%.3f' % (pos_dist[i]) for i in range(pos_dist.shape[0])])
                neg_str = ' '.join(['%.3f' % (neg_dist[i]) for i in range(neg_dist.shape[0])])

                save_path = pjoin(self.opt.eval_dir, 'E%03d.txt' % (epoch))
                with cs.open(save_path, 'w') as f:
                    f.write('Positive Pairs Distance\n')
                    f.write(pos_str + '\n')
                    f.write('Negative Pairs Distance\n')
                    f.write(neg_str + '\n')
        self.writer.close()

