# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import math
import os.path as osp
import random

import numpy as np
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sys

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import ClusterContrastTrainer
from clustercontrast.evaluators import Evaluator, extract_features, extract_imgs
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
from clustercontrast.utils.squeeze_strategy import squeeze_method, label_backward, feature_agents, random_drop_labels, \
    cluster_detection, k1_set, LabelOptic, undersampled_maximum_custer
from tqdm import trange, tqdm
from sklearn.cluster import KMeans


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if iters is None:
        train_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                                  batch_size=batch_size, num_workers=workers, sampler=sampler,
                                  shuffle=not rmgs_flag, pin_memory=True, drop_last=True)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model():
    model = models.create('resnet50', num_features=0, norm=True, dropout=0,
                          num_classes=0, pooling_type='gem')
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True


def main():
    args = parser.parse_args()
    set_seed(args.seed)
    main_worker(args)


def convert_state_dict(state_dict):
    # 创建一个新的状态字典
    new_state_dict = collections.OrderedDict()

    # 遍历原始状态字典中的键值对
    for k, v in state_dict.items():
        # 检查键是否包含 'module.'，如果是，则移除它
        if 'module.' in k:
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def main_worker(args):
    best_mAP = 0
    workers = 4
    height = 256
    width = 128
    num_instances = 16
    eps = 0.6
    k1 = args.k1
    k2 = args.k2
    gamma = args.gamma
    # cams_decay = args.cams_decay
    momentum = 0.1
    print_freq = 10
    eval_step = 20
    temp = 0.05
    p = 0.5
    start_time = time.monotonic()
    cudnn.benchmark = True
    best_load = False

    # record
    map_record = []
    label_recorder = []
    offset_recorder = []
    features_recorder = []

    # sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, height, width, args.batch_size, workers)

    # Create model
    model = create_model()

    # Evaluator
    evaluator = Evaluator(model)

    # testqf
    if best_load:
        s = torch.load('model_best.pth.tar', map_location=torch.device('cpu'))['state_dict']
        s = convert_state_dict(s)
        model.load_state_dict(s)
        mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)

    if args.load_point:
        checkpoint = load_checkpoint(osp.join(args.logs_dir, 'checkpoint.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])

    # Trainer
    trainer = ClusterContrastTrainer(model)
    trainer.memory = ClusterMemory(args.global_momentum, args.temp2).cuda()
    trainer.memory.p=args.p
    if args.load_point:
        # start_epoch = checkpoint['epoch']
        start_epoch = 80
        best_mAP = checkpoint['best_mAP']
        print(f'tranning from {start_epoch}, best {best_mAP}')
    else:
        start_epoch = 0
    # Optimizer
    lr = 0.00035
    params = [{"params": [value], 'initial_lr': lr} for _, value in model.named_parameters() if value.requires_grad]
    for _, value in trainer.cam_task.named_parameters():
         if value.requires_grad:
            params.append({"params": [value], 'initial_lr': lr})
    
    # 0.00035 ,'initial_lr':0.0000035
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.step, gamma=0.1,
                                                        last_epoch=start_epoch - 1)
    
    # Cams_Optimizer
    cams_lr = 0.00035
    cams_params = [{"params": [value], 'initial_lr': lr} for _, value in trainer.cam_task.named_parameters() if value.requires_grad]
    # 0.00035 ,'initial_lr':0.0000035
    cam_optimizer = None
    # cams_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(cam_optimizer, milestones=args.step, gamma=0.1,
                                                        # last_epoch=start_epoch - 1)

    # training
    all_epochs = args.epochs
    bar = tqdm(range(start_epoch, all_epochs), desc='Training start', unit='e', maxinterval=300)
    for epoch in bar:
        # pseudo labels
        with torch.no_grad():
            global_loader = get_test_loader(dataset, height, width,
                                            args.batch_size, workers, testset=sorted(dataset.train))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            f, _ = extract_features(model, global_loader, print_freq=50)
            fs_list = []
            cams = []
            for fp, _, cam in sorted(dataset.train):
                fs_list.append(f[fp])
                cams.append(cam)
            cams = torch.tensor(cams)
            fs = torch.stack(fs_list, 0)
            # torch.save(fs,'msmtfs.pth')
            # torch.save(cams, 'msmtcams.pth')
            ###  CA
            fs, offset = cams_offset(fs, cams)
            # fs = cam_ln(fs, cams)
            # fs = outdim_reduce(fs)
            fs = F.normalize(fs)
            # fs = F.normalize(fs)

            rerank_dist = compute_jaccard_distance(fs, k1=k1, k2=k2, print_flag=False)
            plabels = cluster.fit_predict(rerank_dist)

            # process plabel
            plabels = plabels if isinstance(plabels, torch.Tensor) else torch.tensor(plabels)

        # rename plabel
        pseudo_labels, features = plabels, fs

        # centers
        centers = F.normalize(generate_cluster_features(pseudo_labels, features))
        expand_centers = centers.unsqueeze(1).expand(-1, num_instances, 2048)

        # create pseudo dataset
        pseudo_labeled_dataset = []
        # cams = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        # Count the number of clusters
        num_cluster = torch.max(pseudo_labels) + 1
        num_i = len(torch.where(pseudo_labels != -1)[0])

        trainer.memory.register_buffer('centers', centers.cuda())
        trainer.memory.register_buffer('excenters', expand_centers.cuda())

        # loader
        train_loader = get_train_loader(dataset, height, width,
                                        args.batch_size, workers, num_instances, iters,
                                        trainset=pseudo_labeled_dataset)
        train_loader.new_epoch()

        # online
        trainer.train(epoch, train_loader, optimizer, cam_optimizer,
                      print_freq=print_freq, num_clusters=num_cluster, num_i=num_i, p=args.p)

        lr_scheduler.step()
        # cams_lr_scheduler.step()

        # add result
        label_recorder.append(pseudo_labels)
        offset_recorder.append(offset)

        # output
        if (epoch + 1) % eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            map_record.append(mAP)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            bar.write('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                      format(epoch, mAP, best_mAP, ' *' if is_best else ''))
    # add final result
    features_recorder.append(features)

    # best map
    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

    torch.save(features_recorder, 'final_features.pt')
    torch.save(label_recorder, 'labels.pt')
    torch.save(offset_recorder, 'offset_recorder.pt')
    torch.save(map_record, 'map_record.pt')
    torch.save(trainer.loss_record, 'loss_decline.pt')


# generate new dataset and calculate cluster centers
@torch.no_grad()
def generate_cluster_features(labels, features):
    centers = collections.defaultdict(list)
    for i, label in enumerate(labels):
        if label == -1:
            continue
        centers[labels[i].item()].append(features[i])

    centers = [
        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
    ]

    centers = torch.stack(centers, dim=0)
    return centers


# @torch.no_grad()
# def generate_cluster_features(labels, features):
#     centers = []
#     for i in torch.unique(labels)[1:]:
#         idxs = torch.where(labels == i)[0]
#         # centers.append(features[torch.where(labels == i)].mean(0))
#         centers.append(features[[idxs]][random.randint(0, len(idxs) - 1)])
#     centers = torch.stack(centers, dim=0)
#     return centers

def cam_ln(features, cams):
    cams = cams if isinstance(cams, torch.Tensor) else torch.tensor(cams).long()
    uniq_cam, counts = torch.unique(cams, return_counts=True)
    cams_center = []
    for i in uniq_cam:
        features[torch.where(cams==i)] = F.layer_norm(features[torch.where(cams==i)],[2048])
    return features

def signal_to_noise_ratio(labels):
    noise = len(torch.where(labels == -1)[0])
    return noise / len(labels)


def cams_offset(features, cams):
    cams = cams if isinstance(cams, torch.Tensor) else torch.tensor(cams).long()
    uniq_cam = torch.unique(cams)
    # cams_num = torch.max(cams) + 1
    cams_center = []
    for i in uniq_cam:
        cams_center.append(features[torch.where(cams == i)].mean(0))
    cams_center = torch.stack(cams_center)
    global_center = torch.mean(cams_center, 0)
    offset = cams_center - global_center
    expand_offset = offset[cams]
    features = features - expand_offset
    return features, offset


def labels_map(labels):
    labels = labels if isinstance(labels, torch.Tensor) else torch.tensor(labels)
    anchor = labels.unsqueeze(1).expand(len(labels), len(labels))
    other = labels.unsqueeze(0).expand(len(labels), len(labels))
    mask = torch.eq(anchor, other)
    return mask.numpy()


def jdist_balance(jdist, label=None, mask=None):
    if mask is None:
        mask = labels_map(label)
    p_mean = jdist[mask].mean()
    n_mean = jdist[~mask].mean()

    jdist[mask] += (n_mean - p_mean) / 2
    jdist[~mask] += (p_mean - n_mean) / 2
    pos_bool = (jdist < 0)
    jdist[pos_bool] = 0.0
    return jdist


def jdist_ema(global_jdist, jdist, tau):
    # tau = 0.1
    jdist = global_jdist * tau + jdist * (1 - tau)
    pos_bool = (jdist < 0)
    jdist[pos_bool] = 0.0
    return jdist


def jdist_offset(jdist, cam_mask, tau):
    jdist[cam_mask] = 1
    pos_bool = (jdist < 0)
    jdist[pos_bool] = 0.0
    return jdist


def outdim_reduce(features):
    std = torch.std(features, dim=0)
    mean = torch.mean(features, dim=0)

    for i in features:
        temp = torch.abs(i - mean) / torch.abs(std)
        i[torch.where(temp > 4)] = 0
    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test for new method")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)

    # optimizer
    parser.add_argument('-e', '--epochs', type=int, default=60)
    parser.add_argument('-i', '--iters', type=int, default=200)
    parser.add_argument('-s', '--step', nargs='+', type=int, default=[20, 20])

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--global-momentum', type=float, default=0.1)
    parser.add_argument('--tau', type=float, default=2)
    # parser.add_argument('--use-global', action='store_false')

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/mnt/Datasets/ReID-data')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'examples/logs'))

    parser.add_argument('--k1', type=int, default=30)
    parser.add_argument('--k2', type=int, default=6)
    parser.add_argument('--gamma', type=float, default=2)
    parser.add_argument('--temp2', type=float, default=0.05)
    parser.add_argument('--load-point', action='store_true')
    parser.add_argument('--p', type=float)
    main()
