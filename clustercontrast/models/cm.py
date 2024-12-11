import collections
import math
import random

import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd


class CM_M(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for y in torch.unique(targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * \
                              F.normalize(inputs[torch.where(targets == y)].mean(dim=0))
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cmm(inputs, indexes, features, momentum=0.5):
    return CM_M.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


# class CM_Hard(autograd.Function):
#
#     @staticmethod
#     def forward(ctx, inputs, targets, features, momentum):
#         ctx.features = features
#         ctx.momentum = momentum
#         ctx.save_for_backward(inputs, targets)
#         outputs = inputs.mm(ctx.features.t())
#
#         return outputs
#
#     @staticmethod
#     def backward(ctx, grad_outputs):
#         inputs, targets = ctx.saved_tensors
#         grad_inputs = None
#         if ctx.needs_input_grad[0]:
#             grad_inputs = grad_outputs.mm(ctx.features)
#
#         for y in torch.unique(targets):
#             idxs = torch.where(targets == y)[0]
#             dist = ctx.features[y].unsqueeze(0).mm(inputs[idxs].t())
#             idx = idxs[torch.argmin(dist.squeeze())]
#             ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * inputs[idx]
#             ctx.features[y] /= ctx.features[y].norm()
#
#         return grad_inputs, None, None, None
#
#
# def cm_hard(inputs, indexes, features, momentum=0.5):
#     return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CMA(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum, temp):
        ctx.features = features
        ctx.momentum = momentum
        ctx.temp = temp
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())
        # outputs = torch.einsum('ij,xyj->ixy', inputs, features)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
            # grad_inputs, grad_features = torch.einsum_backward('ij,xyj->ixy', grad_outputs, ctx.features, inputs)

        for y in torch.unique(targets):
            idxs = torch.where(targets == y)[0]
            dist = ctx.features[y].unsqueeze(0).mm(inputs[idxs].t())
            idx = idxs[torch.argmin(dist.squeeze())]
            new_center = F.softmax((inputs[idx].unsqueeze(0).mm(inputs[idxs].t()))/ctx.temp, dim=-1).mm(inputs[idxs])
            new_center /= new_center.norm()
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * new_center
            ctx.features[y] /= ctx.features[y].norm()
        # for y in torch.unique(targets):
        #     idxs = torch.where(targets == y)[0]
        #     ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * inputs[idxs]
        #     ctx.features[y] /= F.normalize(ctx.features[y], dim=-1)

        return grad_inputs, None, None, None, None


def cm_a(inputs, indexes, features, momentum=0.5, temp=0.05):
    return CMA.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device), temp)


class CMR(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = torch.einsum('ij,xyj->ixy', inputs, features)
        # outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = torch.einsum('ixy,xyj->ij', grad_outputs, ctx.features)
            # grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for y in torch.unique(targets):
            idxs = torch.where(targets == y)[0]
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * inputs[
                idxs[random.randint(0, len(idxs) - 1)]]
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cmr(inputs, indexes, features, momentum=0.5):
    return CMR.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, momentum=0.1, temp=0.05):
        super(ClusterMemory, self).__init__()

        self.cluster_idx_list = None
        self.count = None
        self.momentum = momentum
        self.temp = temp
        self.p=None
        # self.emb = nn.Embedding(16, 4, scale_grad_by_freq=True)
        # self.proj = nn.DataParallel(MLP(2048+4,1024,2048))

    def forward(self, inputs, idxs, targets, cams):
        with torch.no_grad():
            # print(targets)
            expand_centers = self.excenters[targets.reshape(-1,16)[:,0]].clone().reshape(-1,2048)
            all_centers = self.excenters.clone().reshape(-1,2048)
            for i in torch.unique(targets):
                self.excenters[i] = F.normalize(self.excenters[i] * 0.1 + 0.9 * inputs[torch.where(targets == i)], dim=-1)
                # inputs[torch.where(targets == i)]
                # F.normalize(self.excenters[i] * 0.1 + 0.9 * inputs[torch.where(targets == i)], dim=-1)



        # 交叉熵损失
        # indices = torch.randperm(16)
        #
        # inputs = inputs.reshape(-1, 16, 2048)
        # inputs = inputs + inputs[:,indices,:]
        # inputs = inputs.reshape(-1, 2048)
        # inputs = F.normalize(inputs, dim=1)

        # c_loss = F.cosine_similarity(inputs, offset_f)
        # c_loss = 2 * (1 - c_loss)

        outputs = cm_hard(inputs, targets, self.centers, self.momentum)
        # outputs = cm_a(inputs, targets, self.centers, self.momentum, self.temp)
        # outputs = torch.einsum('nj,cdj->ncd', inputs, centers)
        outputs = outputs / 0.05
        # print(outputs.shape)
        nce_loss = F.cross_entropy(outputs, targets)
        # nce_loss = nce_loss

        # std_loss = torch.std(inputs, dim=0)
        # std_loss = torch.max(std_loss)


        tau = 0.05
        logits1 = inputs.mm(expand_centers.t())
        logits2 = inputs.mm(all_centers.t())
        logits1 /= tau
        logits2/=tau
        logits1 = torch.exp(logits1)
        logits2 = torch.exp(logits2)
        l2 = logits1.sum(-1)/logits2.sum(-1)
        l2 = -torch.log(l2)
        l2 = l2.mean()
        # print(l2.item())
        ### 来
        loss = nce_loss + l2
        # loss = nce_loss

        # logits = cm_hard(inputs, idxs, self.features, 0.1)
        # logits /= 1
        # logits = F.softmax(logits, dim=-1)
        # f_loss = -torch.log((logits*feature_filter(label_mask, ~cams_mask)).sum(-1))
        # f_loss = f_loss.mean()

        # with torch.no_grad():
        #     for y in torch.unique(targets):
        #         idxs = torch.where(targets == y)[0]
        #         self.centers[y] = 0.1 * self.centers[y] + (1. - 0.1) * inputs[idxs]
        #     self.centers /= F.normalize(self.centers[y], dim=-1)
        return loss

        # return loss + 2 * f_loss + 2 * d_loss

    def reset_count(self):
        self.count = torch.ones(self.features.shape[0]).cuda()

    def label_mapping(self, label):
        anchor = label.unsqueeze(1).expand(len(label), len(label))
        other = label.unsqueeze(0).expand(len(label), len(label))
        mask = torch.eq(anchor, other)
        self.register_buffer('label_mask', mask.cuda())
        self.register_buffer('label', label.cuda())
        # one
        # cluster_idxs = torch.zeros_like(self.label).bool()
        # none one
        cluster_idxs = torch.ones_like(self.label).bool()

        self.register_buffer('cluster_idxs', cluster_idxs.cuda())
        self.cluster_idx_list = []
        for i in torch.unique(self.label):
            c_idxs = torch.where(self.label == i)[0]
            self.cluster_idx_list.append(c_idxs)
            # one
            # self.cluster_idxs[c_idxs[0]] = True

    def cam_mapping(self, label):
        anchor = label.unsqueeze(1).expand(len(label), len(label))
        other = label.unsqueeze(0).expand(len(label), len(label))
        mask = torch.eq(anchor, other)
        self.register_buffer('cam_mask', mask.cuda())
        self.register_buffer('cam', label.cuda())


def feature_filter(main_mask, filter_masks):
    if not isinstance(filter_masks, list):
        filter_masks = [filter_masks]
    for i in filter_masks:
        mask = main_mask & i
        errors = torch.all(~mask, dim=1)
        mask[errors] = main_mask[errors]
    return mask


# class ClusterMemory(nn.Module, ABC):
#     def __init__(self):
#         super(ClusterMemory, self).__init__()
#         # self.num_features = num_features
#         # self.num_samples = num_samples
#
#         self.momentum = 0.1
#         self.temp = 0.05
#         self.use_hard = True
#         # self.beta = beta
#
#         # self.register_buffer('features', torch.zeros(num_samples, num_features))
#
#     def forward(self, inputs, _, targets):
#         inputs = F.normalize(inputs, dim=1).cuda()
#         if self.use_hard:
#             outputs = cm_hard(inputs, targets, self.features, self.momentum)
#         else:
#             outputs = cm1(inputs, targets, self.features, self.momentum)
#         outputs = outputs / self.temp
#         nce_loss = F.cross_entropy(outputs, targets)
#         # re_label = relabel(outputs)
#         # re_nce = F.cross_entropy(outputs, re_label)
#         return nce_loss
class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class STORE_R(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)

        return inputs, targets, features

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_input1 = None
        grad_input2 = None
        grad_input3 = None
        if ctx.needs_input_grad[0]:
            grad_input1 = grad_outputs[0]
        if ctx.needs_input_grad[1]:
            grad_input2 = grad_outputs[1]
        if ctx.needs_input_grad[2]:
            grad_input2 = grad_outputs[2]

        for y in torch.unique(targets):
            idxs = torch.where(targets == y)[0]
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * inputs[
                idxs[random.randint(0, len(idxs) - 1)]]
            ctx.features[y] /= ctx.features[y].norm()

        return grad_input1, grad_input2, grad_input3, None


def store_r(inputs, indexes, features, momentum=0.5):
    return STORE_R.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class STORE_H(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)

        return inputs, targets, features

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_input1 = None
        grad_input2 = None
        grad_input3 = None
        if ctx.needs_input_grad[0]:
            grad_input1 = grad_outputs[0]
        if ctx.needs_input_grad[1]:
            grad_input2 = grad_outputs[1]
        if ctx.needs_input_grad[2]:
            grad_input2 = grad_outputs[2]

        for y in torch.unique(targets):
            idxs = torch.where(targets == y)[0]
            dist = ctx.features[y].unsqueeze(0).mm(inputs[idxs].t())
            idx = idxs[torch.argmin(dist.squeeze())]
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * inputs[idx]
            ctx.features[y] /= ctx.features[y].norm()

        return grad_input1, grad_input2, grad_input3, None


def store_h(inputs, indexes, features, momentum=0.5):
    return STORE_R.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class STORE_M(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)

        return inputs, targets, features

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_input1 = None
        grad_input2 = None
        grad_input3 = None
        if ctx.needs_input_grad[0]:
            grad_input1 = grad_outputs[0]
        if ctx.needs_input_grad[1]:
            grad_input2 = grad_outputs[1]
        if ctx.needs_input_grad[2]:
            grad_input2 = grad_outputs[2]

        # momentum update
        # for x, y in zip(inputs, targets):
        #     ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
        #     ctx.features[y] /= ctx.features[y].norm()
        for y in torch.unique(targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * F.normalize(
                inputs[torch.where(targets == y)].mean(0))
            ctx.features[y] /= ctx.features[y].norm()

        return grad_input1, grad_input2, grad_input3, None


def store_m(inputs, indexes, features, momentum=0.5):
    return STORE_R.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# def mask_generate():
#     f_dim=2048
#     size=16
#     rand_i = random.randint(0,128-1)