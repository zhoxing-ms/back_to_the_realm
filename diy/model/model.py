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
from diy.model.noisy_layers import NoisyLinear


class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False, use_noisy=False, std_init=0.5):
        super().__init__()
        self.use_noisy = use_noisy
        self.std_init = std_init
        
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
        
        # CNN层不变，依然使用常规卷积层
        cnn_flatten = [nn.Flatten(), nn.Linear(512, 128), nn.ReLU(inplace=True)]
        self.cnn_layer = cnn_layer1 + cnn_layer2 + cnn_layer3 + cnn_layer4 + cnn_flatten
        self.cnn_model = nn.Sequential(*self.cnn_layer)

        # 创建优势和价值网络
        self.adv_features, self.val_features, self.adv_out, self.val_out = self._build_networks(
            np.prod(state_shape), np.prod(action_shape)
        )

        if softmax:
            self.softmax_layer = nn.Softmax(dim=-1)
        else:
            self.softmax_layer = None

        self.apply(self.init_weights)

    def _build_networks(self, input_dim, output_dim):
        """构建优势和价值网络，根据use_noisy决定使用哪种线性层"""
        if self.use_noisy:
            # 使用NoisyLinear层
            adv_features = nn.Sequential(
                NoisyLinear(input_dim, 256, self.std_init),
                nn.ReLU(inplace=True),
                NoisyLinear(256, 128, self.std_init),
                nn.ReLU(inplace=True)
            )
            
            val_features = nn.Sequential(
                NoisyLinear(input_dim, 256, self.std_init),
                nn.ReLU(inplace=True),
                NoisyLinear(256, 128, self.std_init),
                nn.ReLU(inplace=True)
            )
            
            adv_out = NoisyLinear(128, output_dim, self.std_init) if output_dim else None
            val_out = NoisyLinear(128, 1, self.std_init) if output_dim else None
            
        else:
            # 使用标准Linear层
            adv_features = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True)
            )
            
            val_features = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True)
            )
            
            adv_out = nn.Linear(128, output_dim) if output_dim else None
            val_out = nn.Linear(128, 1) if output_dim else None
            
        return adv_features, val_features, adv_out, val_out

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear) and not self.use_noisy:
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # 对于NoisyLinear，我们在该类内部已经进行了初始化，不需要在这里处理

    def reset_noise(self):
        """重置所有NoisyLinear层的噪声"""
        if not self.use_noisy:
            return
            
        for module in self.adv_features.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
                
        for module in self.val_features.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
                
        if self.adv_out:
            self.adv_out.reset_noise()
            
        if self.val_out:
            self.val_out.reset_noise()

    # Forward inference
    # 前向推理
    def forward(self, s, state=None, info=None):
        feature_vec, feature_maps = s[0], s[1]
        feature_maps = self.cnn_model(feature_maps)

        feature_maps = feature_maps.view(feature_maps.shape[0], -1)

        concat_feature = torch.concat([feature_vec, feature_maps], dim=1)

        adv = self.adv_features(concat_feature)
        val = self.val_features(concat_feature)
        
        if self.adv_out and self.val_out:
            adv = self.adv_out(adv)
            val = self.val_out(val)
            
            # Dueling DQN: Q = V + (A - mean(A))
            q = val + adv - adv.mean(dim=1, keepdim=True)
            
            if self.softmax_layer:
                q = self.softmax_layer(q)
        else:
            q = adv  # 如果没有输出层，直接返回特征
        
        return q, state
