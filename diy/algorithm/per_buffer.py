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
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=0.01):
        """
        Initialize the buffer with hyperparameters
        
        Args:
            capacity: Maximum size of the buffer
            alpha: Controls how much prioritization is used (0 = uniform, 1 = full prioritization)
            beta: Controls importance sampling weights (0 = no correction, 1 = full correction)
            beta_increment: How much to increase beta over time
            epsilon: Small constant to add to priorities to ensure non-zero probability
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use
        self.beta = beta  # Importance sampling factor
        self.beta_increment = beta_increment  # Beta annealing
        self.epsilon = epsilon  # Small constant to avoid zero priority
        self.max_priority = 1.0  # Initial max priority

    def add(self, sample, error=None):
        """
        Add a new experience to the buffer with priority based on TD error
        """
        # Use max priority for new samples to encourage exploration
        priority = self.max_priority if error is None else (abs(error) + self.epsilon) ** self.alpha
        
        # Add sample to buffer with priority
        self.tree.add(priority, sample)

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