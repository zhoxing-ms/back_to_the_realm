#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :per_buffer.py
@Author  :kaiwu
@Date    :2023/7/1 10:37

"""

import numpy as np
import random
from collections import deque


class SumTree:
    """
    A binary sum tree data structure for efficient priority-based sampling.
    Each leaf node contains a priority value, and each internal node contains the sum of its children's values.
    This allows for O(log n) updates and sampling operations.
    """
    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes
        self.tree = np.zeros(2 * capacity - 1)  # Tree array: [internal nodes | leaf nodes]
        self.data_pointer = 0  # Current position to add new data
        self.size = 0  # Current size of buffer
        self.data = np.zeros(capacity, dtype=object)  # Data storage

    def add(self, priority, data):
        """
        Add new data with its priority to the tree
        """
        # Find the leaf index
        tree_index = self.data_pointer + self.capacity - 1
        
        # Update data frame
        self.data[self.data_pointer] = data
        
        # Update tree with new priority
        self.update(tree_index, priority)
        
        # Move pointer to next position
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        # Update size
        if self.size < self.capacity:
            self.size += 1

    def update(self, tree_index, priority):
        """
        Update the priority of a leaf node and propagate changes up the tree
        """
        # Change at leaf node
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # Propagate changes through tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        """
        Get a leaf node based on a value v (priority sum)
        """
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # Reached bottom of tree
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            # Traverse left or right based on value
            if v <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                v -= self.tree[left_child_index]
                parent_index = right_child_index
        
        data_index = leaf_index - (self.capacity - 1)
        
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total_priority(self):
        """
        Return the sum of all priorities (root node value)
        """
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    A replay buffer that samples experiences based on their TD error priority.
    Uses a SumTree for efficient priority-based sampling.
    Supports n-step returns for better long-term reward consideration.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01, n_steps=1, gamma=0.9):
        """
        Initialize the buffer with hyperparameters
        
        Args:
            capacity: Maximum size of the buffer
            alpha: Controls how much prioritization is used (0 = uniform, 1 = full prioritization)
            beta: Controls importance sampling weights (0 = no correction, 1 = full correction)
            beta_increment: How much to increase beta over time
            epsilon: Small constant to add to priorities to ensure non-zero probability
            n_steps: Number of steps for n-step returns calculation
            gamma: Discount factor for future rewards
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use
        self.beta = beta  # Importance sampling factor
        self.beta_increment = beta_increment  # Beta annealing
        self.epsilon = epsilon  # Small constant to avoid zero priority
        self.max_priority = 1.0  # Initial max priority
        
        # N-step returns parameters
        self.n_steps = n_steps
        self.gamma = gamma
        
        # Buffer to store transitions temporarily for n-step returns calculation
        self.n_step_buffer = deque(maxlen=n_steps)
        
    def _get_n_step_info(self, n_step_buffer):
        """
        Calculate the n-step returns information
        
        Returns:
            (first_obs, first_act, reward, last_obs, done)
            where reward is the n-step discounted reward
        """
        # Get the first observation and action
        first_obs = n_step_buffer[0][0]  # obs
        first_act = n_step_buffer[0][1]  # act
        
        # Get the latest observation and done flag
        last_obs = n_step_buffer[-1][2]  # next_obs (_obs)
        done = n_step_buffer[-1][3]      # done flag
        
        # Calculate n-step discounted reward
        n_step_reward = 0
        for i, (_, _, _, rew, _) in enumerate(n_step_buffer):
            # 检查前面的步骤是否有done=True的情况，如果有则停止累积
            if i > 0 and n_step_buffer[i-1][3]:  # 检查前一个transition是否done
                break
            n_step_reward += (self.gamma ** i) * rew
            
        return first_obs, first_act, n_step_reward, last_obs, done
    
    def add(self, sample, error=None):
        """
        Add a new experience to the n-step buffer. When the buffer reaches n_steps, 
        calculate the n-step return and add it to the replay buffer.
        
        Args:
            sample: (obs, act, _obs, rew, done, _obs_legal)
            error: TD error (optional)
        """
        # Extract data from sample
        obs = sample.obs
        act = sample.act
        _obs = sample._obs
        rew = sample.rew
        done = sample.done
        _obs_legal = sample._obs_legal
        
        # Store in n-step buffer
        self.n_step_buffer.append((obs, act, _obs, rew, _obs_legal))
        
        # 如果有终止状态出现，确保所有的n-step buffer都被处理
        # 我们最多处理n_steps个transition或直到遇到done
        while len(self.n_step_buffer) >= 1:
            # 如果buffer长度小于n_steps且当前episode未结束，等待更多transition
            if len(self.n_step_buffer) < self.n_steps and not done:
                break
                
            # 计算n-step returns
            first_obs, first_act, n_step_reward, last_obs, is_done = self._get_n_step_info(self.n_step_buffer)
            
            # 创建一个修改后的sample，包含n-step信息
            from types import SimpleNamespace
            n_step_sample = SimpleNamespace()
            n_step_sample.obs = first_obs
            n_step_sample.act = first_act
            n_step_sample._obs = last_obs
            n_step_sample.rew = n_step_reward
            n_step_sample.done = is_done
            n_step_sample._obs_legal = self.n_step_buffer[-1][4]  # 使用最新状态的合法动作
            
            # 使用最大优先级来鼓励探索
            priority = self.max_priority if error is None else (abs(error) + self.epsilon) ** self.alpha
            
            # 将sample添加到buffer中并携带优先级
            self.tree.add(priority, n_step_sample)
            
            # 处理过的第一个transition从buffer中移除
            self.n_step_buffer.popleft()
            
            # 如果当前状态是终止状态，我们会在下一次迭代时继续处理剩余的transitions
            # 这样可以确保经验不会丢失
            if is_done and len(self.n_step_buffer) > 0:
                done = True  # 确保循环继续
            else:
                break  # 处理完一个n步序列后退出

    def sample(self, batch_size):
        """
        Sample a batch of experiences based on their priorities
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority() / batch_size
        
        # Increase beta over time for more accurate importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Sample from each segment
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            # Sample a value between a and b
            v = random.uniform(a, b)
            
            # Get corresponding experience from tree
            index, priority, sample = self.tree.get_leaf(v)
            
            batch.append(sample)
            indices.append(index)
            priorities.append(priority)
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total_priority()
        weights = (self.tree.size * sampling_probabilities) ** -self.beta
        
        # Normalize weights to be between 0 and 1
        weights = weights / weights.max()
        
        return batch, indices, weights

    def update_priorities(self, indices, errors):
        """
        Update priorities of sampled experiences based on new TD errors
        """
        for idx, error in zip(indices, errors):
            # Calculate new priority based on TD error
            priority = (abs(error) + self.epsilon) ** self.alpha
            
            # Update max priority for new samples
            self.max_priority = max(self.max_priority, priority)
            
            # Update tree with new priority
            self.tree.update(idx, priority)