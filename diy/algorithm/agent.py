#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :agent.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import time
import math
from diy.model.model import Model
from diy.feature.definition import ActData
import numpy as np
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import attached
from diy.config import Config
from diy.algorithm.per_buffer import PrioritizedReplayBuffer


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.act_shape = Config.DIM_OF_ACTION_DIRECTION + Config.DIM_OF_TALENT
        self.direction_space = Config.DIM_OF_ACTION_DIRECTION
        self.talent_direction = Config.DIM_OF_TALENT
        self.obs_shape = Config.DIM_OF_OBSERVATION
        self.epsilon = Config.EPSILON
        self.egp = Config.EPSILON_GREEDY_PROBABILITY
        self.obs_split = Config.DESC_OBS_SPLIT
        self._gamma = Config.GAMMA
        self.lr = Config.START_LR
        self.use_per = True  # Toggle for Prioritized Experience Replay
        self.use_n_step = Config.USE_N_STEP  # 是否使用n步回报
        self.n_steps = Config.N_STEPS if Config.USE_N_STEP else 1  # 如果不使用n-step则默认为1
        self.use_noisy = Config.USE_NOISY  # 是否使用NoisyNet
        self.noisy_std_init = Config.NOISY_STD_INIT  # NoisyNet初始噪声标准差

        self.device = device
        self.pred_model = Model(
            state_shape=self.obs_shape,
            action_shape=self.act_shape,
            softmax=False,
            use_noisy=self.use_noisy,
            std_init=self.noisy_std_init
        )
        self.target_model = Model(
            state_shape=self.obs_shape,
            action_shape=self.act_shape,
            softmax=False,
            use_noisy=self.use_noisy,
            std_init=self.noisy_std_init
        )
        self.pred_model.to(self.device)
        self.target_model.to(self.device)
        self.copy_pred_to_target()
        self.optim = torch.optim.Adam(self.pred_model.parameters(), lr=self.lr)
        self.train_step = 0
        self.predict_count = 0
        self.last_report_monitor_time = 0
        
        # Initialize PER buffer with n-step parameter
        self.per_buffer = PrioritizedReplayBuffer(
            capacity=10000,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
            epsilon=0.01,
            n_steps=self.n_steps,
            gamma=self._gamma
        )

        self.agent_type = agent_type
        self.logger = logger
        self.monitor = monitor

    def __convert_to_tensor(self, data):
        if isinstance(data, list):
            return torch.tensor(
                np.array(data),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            return torch.tensor(
                data,
                device=self.device,
                dtype=torch.float32,
            )

    def __predict_detail(self, list_obs_data, exploit_flag=False):
        batch = len(list_obs_data)
        feature_vec = [obs_data.feature[: self.obs_split[0]] for obs_data in list_obs_data]
        feature_map = [obs_data.feature[self.obs_split[0] :] for obs_data in list_obs_data]
        legal_act = [obs_data.legal_act for obs_data in list_obs_data]
        legal_act = torch.tensor(np.array(legal_act))
        legal_act = (
            torch.cat(
                (
                    legal_act[:, 0].unsqueeze(1).expand(batch, self.direction_space),
                    legal_act[:, 1].unsqueeze(1).expand(batch, self.talent_direction),
                ),
                1,
            )
            .bool()
            .to(self.device)
        )
        pred_model = self.pred_model
        # 明确设置为评估模式
        pred_model.eval()
        
        # 对于NoisyNet模型，每次预测时重置噪声
        if self.use_noisy:
            pred_model.reset_noise()
        
        # NoisyNet不需要epsilon-greedy，因为噪声已经提供了足够的探索
        # 所以当使用NoisyNet时，直接使用模型输出
        use_random = False
        if not self.use_noisy:
            # 常规epsilon-greedy探索策略
            self.epsilon = max(0.1, self.epsilon - self.predict_count / self.egp)
            # 在评估时不需要随机探索
            use_random = not exploit_flag and np.random.rand(1) < self.epsilon

        with torch.no_grad():
            if use_random:
                random_action = np.random.rand(batch, self.act_shape)
                random_action = torch.tensor(random_action, dtype=torch.float32).to(self.device)
                random_action = random_action.masked_fill(~legal_act, 0)
                act = random_action.argmax(dim=1).cpu().view(-1, 1).tolist()
            else:
                feature = [
                    self.__convert_to_tensor(feature_vec),
                    self.__convert_to_tensor(feature_map).view(batch, *self.obs_split[1]),
                ]
                logits, _ = pred_model(feature, state=None)
                
                # 确保所有掩码外的动作概率为负无穷，防止选择非法动作
                min_val = float('-inf')
                logits = logits.masked_fill(~legal_act, min_val)
                
                # 如果所有动作都是非法的(极端情况)，随机选择一个动作
                if (logits == min_val).all():
                    self.logger.warning("All actions are illegal, selecting random action")
                    act = np.random.randint(0, self.act_shape, size=(batch, 1)).tolist()
                else:
                    act = logits.argmax(dim=1).cpu().view(-1, 1).tolist()

        format_action = [[instance[0] % self.direction_space, instance[0] // self.direction_space] for instance in act]
        self.predict_count += 1
        return [ActData(move_dir=i[0], use_talent=i[1]) for i in format_action]

    @predict_wrapper
    def predict(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=False)

    @exploit_wrapper
    def exploit(self, list_obs_data):
        return self.__predict_detail(list_obs_data, exploit_flag=True)

    @learn_wrapper
    def learn(self, list_sample_data):
        # 如果没有数据，直接返回
        if not list_sample_data or len(list_sample_data) == 0:
            return
            
        # Standard experience replay if PER is disabled
        if not self.use_per:
            t_data = list_sample_data
            batch = len(t_data)
            # Use uniform weights (all 1.0) for standard experience replay
            weights = torch.ones(batch).to(self.device)
            indices = None  # No indices needed for standard experience replay
        else:
            # Add new samples to PER buffer
            for sample in list_sample_data:
                self.per_buffer.add(sample)
                
            # If buffer is too small, just return
            if self.per_buffer.tree.size < 32:  # Minimum batch size
                return
                
            # Sample batch from PER buffer
            batch_size = min(len(list_sample_data), 64)  # Use a reasonable batch size
            t_data, indices, weights = self.per_buffer.sample(batch_size)
            
            # Convert importance sampling weights to tensor
            weights = torch.FloatTensor(weights).to(self.device)
            
            batch = len(t_data)
            
            # 如果采样失败，直接返回
            if batch == 0:
                return

        # [b, d]
        batch_feature_vec = [frame.obs[: self.obs_split[0]] for frame in t_data]
        batch_feature_map = [frame.obs[self.obs_split[0] :] for frame in t_data]
        batch_action = torch.LongTensor(np.array([int(frame.act) for frame in t_data])).view(-1, 1).to(self.device)

        _batch_obs_legal = torch.tensor(np.array([frame._obs_legal for frame in t_data]))
        _batch_obs_legal = (
            torch.cat(
                (
                    _batch_obs_legal[:, 0].unsqueeze(1).expand(batch, self.direction_space),
                    _batch_obs_legal[:, 1].unsqueeze(1).expand(batch, self.talent_direction),
                ),
                1,
            )
            .bool()
            .to(self.device)
        )

        # Get rewards - these will already be n-step returns when using PER with n-steps
        rew = torch.tensor(np.array([frame.rew for frame in t_data]), device=self.device)
        _batch_feature_vec = [frame._obs[: self.obs_split[0]] for frame in t_data]
        _batch_feature_map = [frame._obs[self.obs_split[0] :] for frame in t_data]
        not_done = torch.tensor(np.array([0 if frame.done == 1 else 1 for frame in t_data]), device=self.device)

        batch_feature = [
            self.__convert_to_tensor(batch_feature_vec),
            self.__convert_to_tensor(batch_feature_map).view(batch, *self.obs_split[1]),
        ]
        _batch_feature = [
            self.__convert_to_tensor(_batch_feature_vec),
            self.__convert_to_tensor(_batch_feature_map).view(batch, *self.obs_split[1]),
        ]

        # 设置模型为训练模式
        self.pred_model.train()
        target_model = getattr(self, "target_model")
        target_model.eval()
        
        # 对于NoisyNet，在训练期间也需要重置噪声
        if self.use_noisy:
            self.pred_model.reset_noise()
            target_model.reset_noise()

        with torch.no_grad():
            q, h = target_model(_batch_feature, state=None)
            q = q.masked_fill(~_batch_obs_legal, float('-inf'))  # 使用负无穷替代最小值
            q_max = q.max(dim=1).values.detach()
            
            # 处理所有动作都被掩码的情况
            invalid_mask = torch.isinf(q_max)
            if invalid_mask.any():
                q_max[invalid_mask] = 0.0

        # When using n-step returns, the rewards (rew) already include the discounted sum of n rewards
        # So we only need to add the discounted max Q-value of the nth next state
        # The discount factor for the nth state is gamma^n
        if self.use_per and self.use_n_step and self.n_steps > 1:
            # For n-step returns, the discount factor is gamma^n
            target_q = rew + (self._gamma ** self.n_steps) * q_max * not_done
        else:
            # For 1-step returns (standard DQN)
            target_q = rew + self._gamma * q_max * not_done

        self.optim.zero_grad()

        pred_model = getattr(self, "pred_model")
        logits, h = pred_model(batch_feature, state=None)
        
        # Update priorities in buffer if using PER
        if self.use_per and indices is not None:
            # Calculate TD errors
            td_errors = torch.abs(target_q - logits.gather(1, batch_action).view(-1)).detach().cpu().numpy()
            self.per_buffer.update_priorities(indices, td_errors)
        
        # Apply importance sampling weights to loss if using PER, otherwise use standard loss
        elementwise_loss = torch.square(target_q - logits.gather(1, batch_action).view(-1))
        
        # 添加梯度裁剪，防止梯度爆炸
        max_loss = 100.0
        elementwise_loss = torch.clamp(elementwise_loss, 0, max_loss)
        
        loss = (elementwise_loss * weights).mean()
        loss.backward()

        pred_model_grad_norm = torch.nn.utils.clip_grad_norm_(pred_model.parameters(), 1.0)
        self.optim.step()

        self.train_step += 1
        if(self.train_step % Config.TARGET_UPDATE_FREQ == 0):
            self.copy_pred_to_target()

        value_loss = loss.detach().item()
        q_value = target_q.mean().detach().item()
        reward = rew.mean().detach().item()

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "value_loss": value_loss,
                "q_value": q_value,
                "reward": reward,
                "diy_1": pred_model_grad_norm,
                "diy_2": self.n_steps if self.use_n_step else 0,  # Report n-steps only if enabled
                "diy_3": 1 if self.use_noisy else 0,  # Report NoisyNet status
                "diy_4": self.epsilon if not self.use_noisy else 0,  # Report epsilon only if not using NoisyNet
                "diy_5": 0,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now

    def copy_pred_to_target(self):
        self.target_model.load_state_dict(self.pred_model.state_dict())

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.pred_model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)

        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.pred_model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")
