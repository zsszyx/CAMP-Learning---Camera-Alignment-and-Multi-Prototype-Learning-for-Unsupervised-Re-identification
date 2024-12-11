import random

from .faiss_utils import index_init_cpu
from .faiss_rerank import k_reciprocal_neigh
import torch
import time
import numpy as np
import faiss
from sklearn.cluster import OPTICS
from torch.nn import functional as F


def squeeze_method(target_features, alpha):
    end = time.time()

    temp_labels = k_squeeze(target_features, alpha)
    temp_features = squeeze_features(target_features, temp_labels)
    print('squeeze cost: {:.2f}'.format(time.time()-end))
    return temp_features, temp_labels


def k_squeeze(features, alpha=0.5):
    features = torch.nn.functional.normalize(features)
    temp_labels = torch.full([features.shape[0]], -1)
    dist = 1 - features.mm(features.t())
    upper_idxs = torch.triu_indices(dist.shape[0], dist.shape[0], offset=1)
    benchmark = dist[upper_idxs[0], upper_idxs[1]]
    p_mean = benchmark.mean()
    p = (p_mean-benchmark.min())
    threshold = p_mean - p*alpha
    for idx, label in enumerate(temp_labels):
        if label == -1:
            pl = torch.max(temp_labels)+1
            idxs = dist_search(dist, idx, threshold)
            if len(idxs) > 1:
                temp_labels[idxs] = pl
                temp_labels[idx] = pl
    return temp_labels


def squeeze_features(features, temp_labels):
    uniq_labels = torch.unique(temp_labels[torch.where(temp_labels != -1)])
    fs = []
    for i in uniq_labels:
        indices = torch.where(temp_labels == i)[0]  # 获取所有满足条件的索引
        random_index = random.randint(0, len(indices)-1) # 生成随机排列的索引并选择第一个索引
        selected_element = features[indices[random_index]]
        fs.append(selected_element)
    fs = torch.stack(fs, dim=0)
    return fs


def label_backward(source, target):
    # plabel, templ
    for i, label in enumerate(target):
        target[i] = source[label]
    return target


def dist_search(dist_mat, i, threshold):
    target = np.array([i])
    new_idxs = torch.where(dist_mat[[i]] < threshold)[1]
    result = np.setdiff1d(new_idxs, target)
    while result.shape[0]!=0:
        target = np.append(target, result)
        new_idxs = torch.where(dist_mat[result] < threshold)[1]
        result = np.setdiff1d(new_idxs, target)
    return target

#
# def feature_agents(inherit_labels, features, alpha):
#     if isinstance(inherit_labels, np.ndarray):
#         inherit_labels = torch.tensor(inherit_labels)
#     features = torch.nn.functional.normalize(features)
#     uniq_labels = torch.unique(inherit_labels[torch.where(inherit_labels != -1)])
#     intra_dists = []
#     temp_labels = torch.full([features.shape[0]], -1)
#     dist = 1 - features.mm(features.t())
#
#     for i in uniq_labels:
#         idxs = torch.where(inherit_labels == i)[0]
#         intra_dist = dist[idxs]
#         intra_dist = intra_dist[:, idxs]
#         upper_idxs = torch.triu_indices(intra_dist.shape[0], intra_dist.shape[0], offset=1)
#         upper = intra_dist[upper_idxs[0], upper_idxs[1]]
#         p_mean = upper.mean()
#         intra_dists.append(p_mean)
#     p = min(intra_dists)*1
#
#     for idx, label in enumerate(temp_labels):
#         if label == -1:
#             pl = torch.max(temp_labels)+1
#             idxs = dist_search(dist, idx, p)
#             temp_labels[idxs] = pl
#     features = squeeze_features(features, temp_labels)
#     return features, temp_labels


# def feature_agents2(inherit_labels, features, alpha):
#     if isinstance(inherit_labels, np.ndarray):
#         inherit_labels = torch.tensor(inherit_labels)
#     features = torch.nn.functional.normalize(features)
#     uniq_labels = torch.unique(inherit_labels[torch.where(inherit_labels != -1)])
#
#     # inherit
#     # for i in uniq_labels:
#     #     idxs = torch.where(inherit_labels == i)[0]
#     #
#     #     fs = features[idxs]
#     #     center = fs.mean(dim=0, keepdim=True)
#     #     center = center / center.norm()
#     #     dist = 1 - fs.mm(center.t()).squeeze()
#     #     p = dist.mean() * alpha
#     #     ps_idx = torch.where(dist <= p)[0]
#     #     ns_idx = torch.where(dist > p)[0]
#     #
#     #     inherit_labels[idxs[ps_idx]] = i
#     #     inherit_labels[idxs[ns_idx]] = -1
#
#     p = []
#     dists = []
#     for i in uniq_labels:
#         idxs = torch.where(inherit_labels == i)[0]
#         fs = features[idxs]
#         center = fs.mean(dim=0, keepdim=True)
#         center = center / center.norm()
#         dist = 1 - fs.mm(center.t()).squeeze()
#         dists.append(dist)
#         dist = dist.mean()
#         p.append(dist)
#
#     p = torch.tensor(p).min()
#     for i in uniq_labels:
#         idxs = torch.where(inherit_labels == i)[0]
#         dist = dists[i]
#         ps_idx = torch.where(dist <= p)[0]
#         ns_idx = torch.where(dist > p)[0]
#
#         inherit_labels[idxs[ps_idx]] = i
#         inherit_labels[idxs[ns_idx]] = -1
#
#     # squeeze labels
#     new_labels = torch.full_like(inherit_labels, -1)
#     for idx, (ol, nl) in enumerate(zip(inherit_labels, new_labels)):
#         if ol == -1 and nl == -1:
#             new_labels[idx] = torch.max(new_labels) + 1
#         if ol != -1 and nl == -1:
#             new_labels[torch.where(inherit_labels == ol)] = torch.max(new_labels) + 1
#
#     # squeeze features
#     features = squeeze_features(features, new_labels)
#     features = torch.nn.functional.normalize(features)
#     return features, new_labels


def feature_agents2(inherit_labels, features):
    if isinstance(inherit_labels, np.ndarray):
        inherit_labels = torch.tensor(inherit_labels)

    new_labels = torch.arange(0, inherit_labels.shape[0])
    non_noise_idxs = torch.where(inherit_labels != -1)[0]
    forward_features = features.clone()
    features = torch.nn.functional.normalize(features[non_noise_idxs])
    non_noise_labels = inherit_labels[non_noise_idxs]
    uniq_labels = torch.unique(non_noise_labels)
    centers = []
    for i in uniq_labels:
        idxs = torch.where(non_noise_labels == i)[0]
        fs = features[idxs]
        center = fs.mean(dim=0)
        center = center / center.norm()
        centers.append(center)
    centers = torch.stack(centers)

    dists = features.mm(centers.t())
    for i in uniq_labels:
        idxs = torch.where(non_noise_labels == i)[0] #f [1,3,5,7,9]
        cluster_dists = dists[idxs]
        new_label = torch.argmax(cluster_dists, dim=-1).squeeze()
        inherit_idxs = torch.where(new_label == i) #x<f [1,3,5]
        if len(inherit_idxs[0]) != 0:
            inherit_idxs = non_noise_idxs[inherit_idxs]
            new_labels[inherit_idxs] = new_labels[inherit_idxs[0]].clone()

    # 去重并排序索引序列
    unique_indexs, _ = torch.sort(torch.unique(new_labels))
    # 映射原始索引序列到连续整数序列
    new_labels = torch.searchsorted(unique_indexs, new_labels)

    features = squeeze_features(forward_features, new_labels)
    features = torch.nn.functional.normalize(features)
    return features, new_labels


def feature_agents(dists, indexs, labels, k=0.01):
    labels = torch.tensor(labels)
    flag = torch.ones_like(labels).bool()
    tf_dists = torch.tensor(dists[:, -1])
    k_idxs = torch.tensor(indexs[:, -1])
    sorted_idxs = torch.argsort(tf_dists, descending=True)
    sorted_k_idxs = k_idxs[sorted_idxs]
    t = len(labels) * k
    t = int(t)
    anchor = sorted_idxs[:t]
    positive = sorted_k_idxs[:t]
    for i, j in zip(anchor, positive):
        if flag[i] and flag[j]:
            flag[i] = flag[j] = False
            # if labels[i].item() == labels[j].item() == -1:
            #     labels[i] = labels[j] = torch.max(labels)+1
            # elif labels[i].item() == -1 or labels[j].item() == -1:
            labels[i] = labels[j] = max(labels[i].item(), labels[j].item())
            # elif labels[i].item() != labels[j].item():
            #     labels[i] = labels[j]
    _, labels = torch.unique(labels, return_inverse=True)
    return labels-1


@torch.no_grad()
def random_drop_labels(indexs, labels):# 0.01=81.7
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)
    # data = torch.nn.functional.normalize(features).cpu().numpy()
    # index = faiss.IndexFlatIP(2048)
    # index.add(data)
    # _, indexs = index.search(data, 2)
    indexs = indexs[:, 1]
    indexs = torch.tensor(indexs)

    # klabels = [inherit_labels]
    k_label = labels[indexs]
    uneq_idxs = torch.where((labels != -1) & (k_label != -1) & (k_label != labels))[0]
    # margin_idxs = torch.where(k_label == -1)[0]
    # uneq_idxs = uneq_idxs[~torch.isin(uneq_idxs, margin_idxs)]
    # print(uneq_idxs)
    # p = torch.rand(len(uneq_idxs))
    # uneq_idxs = uneq_idxs[p > temp]
    labels[uneq_idxs] = -1
    _, labels = torch.unique(labels, return_inverse=True)
    labels = labels - 1

    # centers = []
    # for i in torch.unique(labels)[1:]:
    #     centers.append(features[torch.where(labels == i)].mean(0))
    # centers = torch.stack(centers, dim=0)
    # centers = torch.nn.functional.normalize(centers, dim=-1)
    #
    # labels = cluster_detection(centers, features, labels)
    return labels


def cluster_detection(center, features, label):
    # label should with noises
    cos_dist = 1 - features.mm(center.t())
    # len(cos_dist)=len(non_noise_idxs)
    non_noise_idxs = torch.where(label != -1)[0]
    # new_label = label.copy()
    # dists = []
    for i in torch.unique(label[non_noise_idxs]):
        idxs = torch.where(label[non_noise_idxs] == i)[0]
        dist = cos_dist[idxs, i].squeeze()
        mean = torch.mean(dist)
        std = torch.std(dist)
        threshold = 2 * std
        outliers = torch.where(torch.abs(dist - mean) >= threshold)[0]
        label[non_noise_idxs[idxs[outliers]]] = -1
    _, label = torch.unique(label, return_inverse=True)
    return label-1
# @torch.no_grad()
# def kernel_center():


def k1_set(features, labels, temp=0.01):
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)
    data = features.cpu().numpy()
    index = faiss.IndexFlatL2(2048)
    index.add(data)
    dists, indexs = index.search(data, 2)
    tf_dists = torch.tensor(dists[:, -1])
    k_idxs = indexs[:, -1]
    sorted_indices = torch.argsort(tf_dists)
    ks_idxs = k_idxs[sorted_indices]
    t = len(features) * temp
    t = int(t)
    anchor = sorted_indices[:t]
    positive = ks_idxs[:t]

    # labels = torch.full(size=[len(features)], fill_value=-1)

    for e, (i, j) in enumerate(zip(anchor, positive)):
        labels[i] = labels[j] = max(labels[i], labels[j]).item()
    _, labels = torch.unique(labels, return_inverse=True)
    return labels-1


class LabelOptic:
    def __init__(self, anomaly=3):
        self.op = OPTICS(min_samples=2)
        self.anomaly = anomaly

    def anomaly_detection(self, plabels):
        ids, counts = torch.unique(plabels, return_counts=True)
        counts = counts[1:].float()
        threshold = counts.mean() + counts.std() * self.anomaly
        ids = ids[1:]
        # anomaly_id = ids[torch.where(counts > threshold)]
        anomaly_id = [] if torch.max(counts) <= threshold else ids[torch.argmax(counts)].unsqueeze(0)
        # print(anomaly_id)
        return anomaly_id

    def reclustering(self, features, plabels):
        anomaly_id = self.anomaly_detection(plabels)
        dists = {}
        for i in anomaly_id:
            dists[i] = torch.where(plabels == i)[0]
            cluster_features = F.normalize(features[dists[i]]).numpy()
            deep_label = torch.tensor(self.op.fit_predict(cluster_features))
            deep_label[torch.where(deep_label!=-1)] += torch.max(plabels)+1
            plabels[dists[i]] = deep_label
        _, labels = torch.unique(plabels, return_inverse=True)
        return labels - 1


def undersampled_maximum_custer(plabels):
    ids, counts = torch.unique(plabels, return_counts=True)
    counts = counts[1:]
    ids = ids[1:]
    mid = ids[torch.argmax(counts)]
    plabels[torch.where(plabels==mid)] = -1
    _, labels = torch.unique(plabels, return_inverse=True)
    return labels - 1









