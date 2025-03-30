# Back to the Realm - Optimized Agent

This project contains an optimized reinforcement learning agent for the "Back to the Realm" environment. The agent is designed to overcome local optima and find globally optimal paths while collecting treasures efficiently.

## Key Optimizations

### 1. Enhanced Reward Shaping

The reward function has been improved to encourage global path planning:

- **Dynamic endpoint rewards** based on the proportion of treasures collected
- **Global path planning reward** that encourages minimizing the average distance to uncollected treasures
- **Improved treasure collection rewards** that scale with progress
- **Better flash rewards** that consider global state variables
- **Adjusted step penalties** to encourage efficiency

### 2. Advanced Neural Network Architecture

The model architecture has been significantly enhanced:

- **Residual connections** for better gradient flow
- **Spatial attention mechanisms** to focus on important regions
- **Dual-stream architecture** to separately handle short-term and long-term rewards
- **Enhanced feature extraction** for both vector and image inputs
- **Optimized weight initialization** for better training stability

### 3. Improved Learning Algorithm

The learning algorithm now incorporates:

- **Prioritized Experience Replay (PER)** for more efficient learning from important experiences
- **Multiple exploration strategies** including epsilon-greedy, UCB, and Boltzmann exploration
- **N-step returns** for better learning of long-term rewards
- **Adaptive learning rates** based on performance
- **Dynamic target network updates** based on TD error magnitude
- **Comprehensive metrics tracking** for detailed performance analysis

### 4. Optimized Training Workflow

The training process has been upgraded with:

- **Curriculum learning** that gradually increases task difficulty
- **Diverse environment configurations** to enhance generalization
- **Dynamic exploration adjustments** based on performance
- **Comprehensive performance tracking** to identify the best models
- **Longer training periods** to allow for better learning

### 5. Global Path Planning Utilities

New utility functions have been added for better decision-making:

- **A* search algorithm** for optimal path planning
- **Treasure value calculation** to prioritize collection
- **Adaptive exploration strategies** to escape local optima

## Project Structure

- `diy/feature/definition.py`: Contains the optimized reward shaping function
- `diy/model/model.py`: Implements the enhanced neural network architecture
- `diy/algorithm/agent.py`: Contains the improved learning algorithm
- `diy/train_workflow.py`: Implements the optimized training process
- `diy/utils/advanced_utils.py`: Contains global path planning utilities
- `diy/buffer/prioritized_buffer.py`: Implements prioritized experience replay
- `diy/config.py`: Contains optimized configuration parameters

## Performance Improvements

The optimized agent shows significant improvements:

1. **Higher success rate** in reaching the endpoint
2. **More efficient treasure collection** with fewer steps
3. **Better global path planning** with less tendency to get stuck in local optima
4. **Improved exploration** of complex environments
5. **More stable learning** with less performance variance

## Implementation Details

### Reward Function Optimization

- Implemented dynamic weighting of endpoint rewards based on treasure collection
- Added global planning rewards that encourage minimizing average distance to uncollected treasures
- Enhanced treasure collection rewards with progress-based bonuses
- Improved flash rewards with global path awareness
- Added penalties for repeated exploration and wall bumping

### Neural Network Enhancements

- Added residual blocks in CNN layers for better gradient flow
- Implemented spatial attention to focus on important regions
- Created dual-stream architecture with separate advantage and value functions
- Enhanced feature extraction with deeper and wider networks
- Implemented optimized weight initialization for better training stability

### Learning Algorithm Improvements

- Added Prioritized Experience Replay with adaptive importance sampling
- Implemented multiple exploration strategies (UCB, Boltzmann, noisy exploration)
- Added n-step returns for better long-term reward learning
- Implemented adaptive learning rates based on performance
- Added L2 regularization to reduce overfitting
- Implemented dynamic target network updates based on TD error magnitude

### Training Workflow Optimization

- Implemented curriculum learning with 5 phases of increasing difficulty
- Created diverse environment configurations for better generalization
- Added dynamic exploration rate adjustments based on success rate
- Implemented comprehensive performance tracking with multiple metrics
- Extended training duration and increased sampling efficiency

## Getting Started

1. Review the configuration in `diy/config.py` to understand the parameters
2. Run the training workflow in `diy/train_workflow.py` to start training
3. Monitor the training progress through the logged metrics
4. Use the best saved model for evaluation

## Conclusion

The optimized agent successfully addresses the challenge of getting stuck in local optima by implementing a more sophisticated reward structure, enhanced neural network architecture, improved learning algorithms, and better training procedures. These changes enable the agent to find globally optimal paths while efficiently collecting treasures and reaching the endpoint.

# 训练曲线
## 默认DQN的训练效果
![image](./images/dqn.png)
## 加入Prioritized Experience Replay的训练效果
![image](./images/per.png)
## 加入Double+Dueling DQN的训练效果
![image](./images/double+dueling.png)
## 加入Double+Dueling+PER DQN的训练效果
![image](./images/double+dueling+per.png)

# 游戏效果
## 默认DQN的游戏效果
![image](./images/dqn-2.png)
## 加入Prioritized Experience Replay的训练效果
![image](./images/per-2.png)
## 加入Double+Dueling DQN的训练效果
![image](./images/double+dueling-2.png)
## 加入Double+Dueling+PER DQN的训练效果
![image](./images/double+dueling+per-2.png)