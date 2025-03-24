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
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        ]
        cnn_layer2 = [
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        cnn_layer3 = [
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        cnn_layer4 = [
            nn.Conv2d(64, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        ]
        cnn_flatten = [nn.Flatten(), nn.Linear(512, 128), nn.ReLU(inplace=True)]

        self.cnn_layer = cnn_layer1 + cnn_layer2 + cnn_layer3 + cnn_layer4 + cnn_flatten
        self.cnn_model = nn.Sequential(*self.cnn_layer)

        fc_a_layer1 = [nn.Linear(np.prod(state_shape), 256), nn.ReLU(inplace=True)]
        fc_a_layer2 = [nn.Linear(256, 128), nn.ReLU(inplace=True)]
        fc_a_layer3 = [nn.Linear(128, np.prod(action_shape))]

        self.fc_a_layers = fc_a_layer1 + fc_a_layer2

        if action_shape:
            self.fc_a_layers += fc_a_layer3
        if softmax:
            self.fc_a_layers += [nn.Softmax(dim=-1)]

        fc_v_layer1 = [nn.Linear(np.prod(state_shape), 256), nn.ReLU(inplace=True)]
        fc_v_layer2 = [nn.Linear(256, 128), nn.ReLU(inplace=True)]
        fc_v_layer3 = [nn.Linear(128, 1)]

        self.fc_v_layers = fc_v_layer1 + fc_v_layer2

        if action_shape:
            self.fc_v_layers += fc_v_layer3
        if softmax:
            self.fc_v_layers += [nn.Softmax(dim=-1)]

        self.model_v = nn.Sequential(*self.fc_v_layers)
        self.model_a = nn.Sequential(*self.fc_a_layers)

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

        feature_maps = feature_maps.view(feature_maps.shape[0], -1)

        concat_feature = torch.concat([feature_vec, feature_maps], dim=1)

        a = self.model_a(concat_feature)
        v = self.model_v(concat_feature)
        q = a + v - a.mean(dim=1, keepdim =True)
        return q, state
