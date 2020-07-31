#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class SharedNet(nn.Module):
    def __init__(self, config, input_len, input_collapse=True):
        super(SharedNet, self).__init__()
        conv1_num_filter = config["conv1_num_filter"]
        conv2_num_filter = config["conv2_num_filter"]
        conv1_kernel_size = config["conv1_kernel_size"]
        conv2_kernel_size = config["conv2_kernel_size"]
        fc_len = config["fc_len"]
        linear_input = (((input_len - conv1_kernel_size + 1) // 2) - conv2_kernel_size + 1) // 2
        self.input_collapse = input_collapse
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2, stride=2)
        self.conv1 = nn.Conv1d(2, conv1_num_filter, kernel_size=conv1_kernel_size)
        self.conv2 = nn.Conv1d(conv1_num_filter, conv2_num_filter, kernel_size=conv2_kernel_size)
        self.bn1 = nn.BatchNorm1d(conv1_num_filter)
        self.bn2 = nn.BatchNorm1d(conv2_num_filter)
        self.fc = nn.Linear(conv2_num_filter * linear_input, fc_len)

    def forward(self, x, mask_input, mask_conv1, mask_conv2):
        x = x.transpose(1, 2)  # A hack to handle Conv1d input
        # Input can be randomly corrupted with (0, 0) vector to learn missing-values during training. 
        if self.training and self.input_collapse:
            if mask_input is None:
                mask_input = torch.bernoulli(x.data.new(x.data.size()).fill_(random.uniform(0.8, 1)))
            x = x * mask_input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        if mask_conv1 is None:
            mask_conv1 = torch.bernoulli(x.data.new(x.data.size()).fill_(0.5))
        if self.training:
            x = x * mask_conv1
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        if mask_conv2 is None:
            mask_conv2 = torch.bernoulli(x.data.new(x.data.size()).fill_(0.5))
        if self.training:
            x = x * mask_conv2
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x, mask_input, mask_conv1, mask_conv2


class EachNet(nn.Module):
    def __init__(self, config, num_class):
        super(EachNet, self).__init__()
        fc_len = config["fc_len"]
        self.fc = nn.Linear(fc_len, num_class)

    def forward(self, x, mask_fc):
        if mask_fc is None:
            mask_fc = torch.bernoulli(x.data.new(x.data.size()).fill_(0.5))
        if self.training:
            x = x * mask_fc
        x = self.fc(x)
        return F.log_softmax(x, dim=1), mask_fc
