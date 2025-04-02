# 训练曲线
## 默认DQN的训练效果
![image](./images/dqn.png)
## 加入Prioritized Experience Replay的训练效果
![image](./images/per.png)
## 加入Double+Dueling DQN的训练效果
![image](./images/double+dueling.png)
## 加入Double+Dueling+PER DQN的训练效果
![image](./images/double+dueling+per.png)
## 加入N步回报的训练效果
N步回报（N-step Returns）是一种改进的强化学习技术，它使智能体不仅考虑即时奖励，还考虑未来多步的累积奖励。与传统的1步回报相比，N步回报能够让智能体更加注重长期的收益，提高学习效率，特别是在奖励稀疏的环境中。

具体优势：
1. 更快的价值传播：通过直接使用多步奖励的和，价值信息可以更快地传播
2. 减少短视行为：智能体不会过度关注即时奖励，而是考虑到长远利益
3. 提高探索效率：更有效地评估状态价值，有助于探索更有价值的路径

在本项目中，我们使用了3步回报（可在config.py中配置），并结合优先经验回放（PER）一起使用，以获得最佳的训练效果。

## 加入NoisyNet的训练效果
NoisyNet是一种在深度强化学习中用于探索的创新技术。它通过向网络的权重和偏置添加参数化噪声，替代了传统的ε-greedy探索策略。

主要特点和优势：
1. 自适应探索：噪声大小会随着学习的进行而自动调整，比ε-greedy更加灵活
2. 状态依赖探索：在不同状态下可以产生不同程度的探索，更加高效
3. 无需手动调整：不需要设计复杂的探索衰减策略，降低了超参数调优的复杂性
4. 与其他强化学习改进兼容：可以与Dueling架构、PER、n步回报等技术无缝结合

在本项目中，NoisyNet替代了传统的ε-greedy探索策略，通过对网络中的线性层添加因子化高斯噪声实现参数化探索。可以在config.py中通过USE_NOISY开关启用或禁用此功能。

# 游戏效果
## 默认DQN的游戏效果
![image](./images/dqn-2.png)
## 加入Prioritized Experience Replay的训练效果
![image](./images/per-2.png)
## 加入Double+Dueling DQN的训练效果
![image](./images/double+dueling-2.png)
## 加入Double+Dueling+PER DQN的训练效果
![image](./images/double+dueling+per-2.png)