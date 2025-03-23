#!/usr/bin/env python3
# -*- coding:utf-8 -*-


"""
@Project :back_to_the_realm
@File    :model.py
@Author  :kaiwu
@Date    :2022/11/15 20:57

"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
        cnn_layer1 = [
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        cnn_layer2 = [
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        cnn_layer3 = [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        cnn_layer4 = [
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        adaptive_pool = [nn.AdaptiveAvgPool2d((4, 4))]
        flatten_layer = [nn.Flatten()]
        self.cnn_layer = cnn_layer1 + cnn_layer2 + cnn_layer3 + cnn_layer4 + adaptive_pool + flatten_layer
        self.cnn_model = nn.Sequential(*self.cnn_layer)

        fc_layer1 = [nn.Linear(64 * 4 * 4 + state_shape[0], 512), nn.ReLU(inplace=True)]
        fc_layer2 = [nn.Linear(512, 256), nn.ReLU(inplace=True)]
        fc_layer3 = [nn.Linear(256, np.prod(action_shape))]

        self.fc_layers = fc_layer1 + fc_layer2

        if action_shape:
            self.fc_layers += fc_layer3
        if softmax:
            self.fc_layers += [nn.Softmax(dim=-1)]

        self.model = nn.Sequential(*self.fc_layers)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Forward inference
    # 前向推理
    def forward(self, s, state=None, info=None):
        feature_vec, feature_maps = s[0], s[1]
        feature_maps = self.cnn_model(feature_maps)

        concat_feature = torch.cat([feature_vec, feature_maps], dim=1)

        logits = self.model(concat_feature)
        return logits, state
