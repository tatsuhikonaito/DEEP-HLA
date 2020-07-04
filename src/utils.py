#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import torch.nn.functional as F


class Logger(object):
    def __init__(self, fn):
        self.f = open(fn, 'w')

    def log(self, msg, *args, **kwargs):
        msg = msg.format(*args, **kwargs)
        print(msg)
        self.f.write(msg+"\n")


def nll(pred, gt, val=False):
    if val:
        return F.nll_loss(pred, gt, size_average=False)
    else:
        return F.nll_loss(pred, gt)


def get_loss(num_task):
    loss_fn = {}
    for t in range(num_task):
        loss_fn[t] = nll
    return loss_fn


class RunningMetric(object):
    def __init__(self):
        self.accuracy = 0.0
        self.num_updates = 0.0

    def reset(self):
        self.accuracy = 0.0
        self.num_updates = 0.0

    def update(self, pred, gt):
        predictions = pred.data.max(1, keepdim=True)[1]
        self.accuracy += (predictions.eq(gt.data.view_as(predictions)).cpu().sum())
        self.num_updates += predictions.shape[0]

    def get_result(self):
        return {'acc': self.accuracy / self.num_updates}


def get_metrics(task_num):
    met = {}
    for t in range(task_num):
        met[t] = RunningMetric()
    return met
