from __future__ import print_function, absolute_import

import math
import random
import time
from .utils.meters import AverageMeter
from torch import nn
import torch
from tqdm import tqdm
from torch.nn import functional as F
import copy


class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.loss_record=[]
        self.cam_task=nn.Sequential(nn.Linear(4096, 1024), nn.ReLU(), nn.Linear(1024,256), nn.ReLU(), nn.Linear(256,64), nn.ReLU(), nn.Linear(64, 2), nn.Softmax(1)).cuda()
    

    def train(self, epoch, data_loader, optimizer,cam_optimizer, print_freq=10, num_clusters=0, num_i=0, p=0):
        self.encoder.train()

        losses = AverageMeter()

        bar = tqdm(data_loader, desc=f'{epoch}e, {num_clusters}c, {num_i}i', unit='it', miniters=10, maxinterval=30)
        for i, inputs in enumerate(bar):
            # load data

            # process inputs
            inputs, labels, cams, indexes = self._parse_data(inputs)
            # print(inputs.shape)
            sorted_values, sorted_indices = torch.sort(labels)
            cams_judge=cams_judge_label(cams[sorted_indices])
            

            # forward
            # inputs = shuffle_merge(inputs).cuda()
            f_out = self._forward(inputs)
            cos_loss = self.memory(f_out, indexes, labels, cams)
            
            cam_loss = F.cross_entropy(self.cam_task(f_out[sorted_indices].reshape(-1,4096)), cams_judge.cuda())
            




            
            # cam_optimizer.zero_grad()
            # cams_loss.backward()
            # cam_optimizer.step()
            
            
            loss = cos_loss + 0.05*cam_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            if i==0:
                # print(1)
                self.loss_record.append(i)

            # print log
            if (i + 1) % print_freq == 0:
                bar.set_postfix(loss=losses.val, loss_avg=losses.avg)

    def _parse_data(self, inputs):
        imgs, _, pids, cams, indexes = inputs
        return imgs.cuda(), pids.cuda(), cams.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


def shuffle_merge(inputs, instance=16):
    B = inputs.shape[0]
    inputs = inputs.reshape(-1,instance, 3, 256, 128)
    indices = torch.randperm(instance)
    s_inputs = inputs[:, indices].reshape(-1,3, 256, 128).cuda()
    inputs = inputs.reshape(-1,3, 256, 128)
    a = torch.rand(B).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
    return inputs*a+(1-a)*s_inputs

def cams_judge_label(labels):

    # 计算tensor的长度
    length = labels.size(0)

    # 检查长度是否为偶数，如果不是，需要先去掉一个元素以保证长度为偶数
    assert length % 2 == 0, f'cams长度不为偶数:length={length}'


    # 创建一个新的tensor，长度为原长度的一半
    new_labels = torch.zeros(length // 2, dtype=torch.int64)

    # 遍历原始tensor中的元素，并比较两两相邻的元素
    for i in range(length // 2):
    # 比较相邻的两个元素
        if labels[2 * i] == labels[2 * i + 1]:
            new_labels[i] = 1
        else:
            new_labels[i] = 0
    new_labels = new_labels.view(-1)
    # new_labels 现在包含了相邻元素是否相等的标签
    # print(new_labels)
    return new_labels
