#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :train_workflow.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

import time
import numpy as np
import random
from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached


from diy.feature.definition import (
    observation_process,
    action_process,
    sample_process,
    reward_shaping,
)
from conf.usr_conf import usr_conf_check


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]

    # 优化：增加训练轮次以提高全局规划能力
    # Optimization: Increase training epochs to improve global planning capability
    epoch_num = 200000  # 从150000增加到200000，给予更充分的学习时间
    
    # 优化：更高效的采样和更大批量的学习
    # Optimization: More efficient sampling and larger batch learning
    episode_num_every_epoch = 3  # 增加到3，提高样本多样性
    
    # 优化：增加经验序列长度以更好地学习长期依赖
    # Optimization: Increase experience sequence length for better long-term dependency learning
    g_data_truncat = 512  # 增加到512，增强时序信息学习
    
    last_save_model_time = 0
    
    # 优化：引入课程学习策略，从简单到困难
    # Optimization: Introduce curriculum learning strategy, from simple to hard
    curriculum_phase = 0  # 课程学习阶段
    curriculum_thresholds = [500, 3000, 7000, 12000, 20000]  # 课程学习阶段转换点
    
    # 优化：设计更丰富的环境配置用于训练，增强泛化能力和全局规划
    # Optimization: Design more diverse environment configurations for training to enhance generalization and global planning
    usr_conf_templates = [
        # 阶段1：基础训练，少量宝箱，专注于直接路径规划
        # Phase 1: Basic training, few treasures, focus on direct path planning
        {
            "diy": {
                "start": 2,
                "end": 1,
                "treasure_random": 1,
                "talent_type": 1,
                "treasure_num": 3,  # 少量宝箱
                "max_step": 1000,   # 较短步数限制
            }
        },
        # 阶段2：中级训练，中等宝箱数量，平衡直接性和收集性
        # Phase 2: Intermediate training, medium number of treasures, balance directness and collection
        {
            "diy": {
                "start": 2,
                "end": 1,
                "treasure_random": 1,
                "talent_type": 1,
                "treasure_num": 7,  # 中等宝箱数量
                "max_step": 1500,   # 中等步数限制
            }
        },
        # 阶段3：高级训练，大量宝箱，注重全局收集策略
        # Phase 3: Advanced training, large number of treasures, focus on global collection strategy
        {
            "diy": {
                "start": 2,
                "end": 1,
                "treasure_random": 1,
                "talent_type": 1,
                "treasure_num": 12,  # 大量宝箱
                "max_step": 2000,    # 较长步数限制
            }
        },
        # 固定宝箱配置：特殊路径训练
        # Fixed treasure configuration: Special path training
        {
            "diy": {
                "start": 2,
                "end": 1,
                "treasure_id": [3, 4, 5, 6, 7, 8, 9],
                "treasure_random": 0,
                "talent_type": 1,
                "max_step": 2000,
            }
        },
        # 极限全收集测试：全部宝箱
        # Extreme full collection test: All treasures
        {
            "diy": {
                "start": 2,
                "end": 1,
                "treasure_random": 1,
                "talent_type": 1,
                "treasure_num": 13,  # 全部宝箱
                "max_step": 2500,    # 较长步数限制
            }
        },
        # 随机起点终点配置：提高适应性
        # Random start-end configuration: Improve adaptability
        {
            "diy": {
                "start": random.choice([2, 3, 4, 5]),
                "end": random.choice([1, 10, 11, 12]),
                "treasure_random": 1,
                "talent_type": 1,
                "treasure_num": 8,
                "max_step": 2000,
            }
        },
        # 无宝箱配置：纯路径寻找训练
        # No-treasure configuration: Pure path finding training
        {
            "diy": {
                "start": 2,
                "end": 1,
                "treasure_random": 1,
                "talent_type": 1,
                "treasure_num": 0,  # 无宝箱
                "max_step": 800,    # 更短步数限制
            }
        },
    ]

    # 跟踪训练指标
    # Track training metrics
    best_avg_reward = -float('inf')
    training_start_time = time.time()
    episode_success_rate = 0.0  # 成功率追踪
    success_count = 0
    total_episodes = 0
    recent_rewards = []  # 最近的奖励列表
    
    # 指标计算和记录
    # Metrics calculation and recording
    treasures_collected_history = []  # 宝箱收集历史
    steps_to_complete_history = []    # 完成步数历史
    avg_treasures_collected = 0
    avg_steps_to_complete = 0
    
    for epoch in range(epoch_num):
        epoch_total_rew = 0
        data_length = 0
        epoch_treasures_collected = 0
        epoch_steps_to_complete = 0
        epoch_episodes_completed = 0
        
        # 优化：根据课程学习阶段选择适当的环境配置
        # Optimization: Select appropriate environment configuration based on curriculum learning phase
        if epoch < curriculum_thresholds[0]:
            # 阶段1：简单配置，专注于基础移动和少量宝箱
            # Phase 1: Simple configuration, focus on basic movement and few treasures
            curriculum_phase = 0
            usr_conf_indices = [0, 6]  # 基础训练和无宝箱配置
        elif epoch < curriculum_thresholds[1]:
            # 阶段2：添加更多宝箱，开始学习集合规划
            # Phase 2: Add more treasures, start learning collection planning
            curriculum_phase = 1
            usr_conf_indices = [0, 1, 6]  # 基础训练、中级训练和无宝箱配置
        elif epoch < curriculum_thresholds[2]:
            # 阶段3：增加复杂性，引入固定宝箱配置
            # Phase 3: Increase complexity, introduce fixed treasure configurations
            curriculum_phase = 2
            usr_conf_indices = [1, 3, 6]  # 中级训练、固定宝箱和无宝箱配置
        elif epoch < curriculum_thresholds[3]:
            # 阶段4：高级训练，专注于全局收集策略
            # Phase 4: Advanced training, focus on global collection strategy
            curriculum_phase = 3
            usr_conf_indices = [1, 2, 3, 5]  # 中级训练、高级训练、固定宝箱和随机起终点
        else:
            # 阶段5：全面训练，包括极限全收集
            # Phase 5: Comprehensive training, including extreme full collection
            curriculum_phase = 4
            usr_conf_indices = [2, 3, 4, 5]  # 高级训练、固定宝箱、极限全收集和随机起终点
        
        # 在当前课程阶段的配置中随机选择一个
        # Randomly select a configuration from the current curriculum phase
        usr_conf_idx = random.choice(usr_conf_indices)
        usr_conf = usr_conf_templates[usr_conf_idx]
        
        # 特殊处理：随机起终点配置需要每次重新生成
        # Special handling: Random start-end configuration needs to be regenerated each time
        if usr_conf_idx == 5:
            usr_conf["diy"]["start"] = random.choice([2, 3, 4, 5])
            usr_conf["diy"]["end"] = random.choice([1, 10, 11, 12])
            while usr_conf["diy"]["start"] == usr_conf["diy"]["end"]:
                usr_conf["diy"]["end"] = random.choice([1, 10, 11, 12])
        
        # 检查配置有效性
        # Check configuration validity
        valid = usr_conf_check(usr_conf, logger)
        if not valid:
            logger.error(f"usr_conf_check return False, please check")
            continue
        
        # 优化：动态调整探索率，根据训练进度和性能
        # Optimization: Dynamically adjust exploration rate based on training progress and performance
        if hasattr(agent, 'epsilon') and episode_success_rate < 0.8:
            # 如果成功率低，增加探索
            # If success rate is low, increase exploration
            agent.epsilon = max(0.1, agent.epsilon + 0.02)
        
        # 运行episode并收集数据
        # Run episodes and collect data
        episode_treasure_counts = []
        episode_step_counts = []
        
        for g_data in run_episodes(episode_num_every_epoch, env, agent, g_data_truncat, usr_conf, logger):
            data_length += len(g_data)
            total_rew = sum([i.rew for i in g_data])
            epoch_total_rew += total_rew
            
            # 优化：处理经验重播缓冲区的填充
            # Optimization: Handle experience replay buffer filling
            agent.learn(g_data)
            
            # 记录完成情况和宝箱收集情况
            # Record completion and treasure collection information
            if len(g_data) > 0 and g_data[-1].done:
                episode_treasures, episode_steps = g_data[-1].ret, len(g_data)
                episode_treasure_counts.append(episode_treasures)
                episode_step_counts.append(episode_steps)
                
                # 更新完成统计
                # Update completion statistics
                if episode_treasures > 0:  # 正奖励表示成功完成
                    epoch_episodes_completed += 1
                    epoch_treasures_collected += episode_treasures
                    epoch_steps_to_complete += episode_steps
            
            g_data.clear()

        # 计算平均步奖励
        # Calculate average step reward
        avg_step_reward = 0
        if data_length:
            avg_step_reward = epoch_total_rew/data_length
            avg_step_reward_str = f"{avg_step_reward:.2f}"
            recent_rewards.append(avg_step_reward)
            if len(recent_rewards) > 50:
                recent_rewards.pop(0)

        # 优化：更新成功率追踪
        # Optimization: Update success rate tracking
        if epoch_episodes_completed > 0:
            success_count += epoch_episodes_completed
            total_episodes += episode_num_every_epoch
            episode_success_rate = success_count / max(1, total_episodes)
            
            # 更新宝箱和步数统计
            # Update treasure and step statistics
            avg_epoch_treasures = epoch_treasures_collected / epoch_episodes_completed
            avg_epoch_steps = epoch_steps_to_complete / epoch_episodes_completed
            
            treasures_collected_history.append(avg_epoch_treasures)
            steps_to_complete_history.append(avg_epoch_steps)
            
            if len(treasures_collected_history) > 50:
                treasures_collected_history.pop(0)
            if len(steps_to_complete_history) > 50:
                steps_to_complete_history.pop(0)
                
            avg_treasures_collected = np.mean(treasures_collected_history)
            avg_steps_to_complete = np.mean(steps_to_complete_history)

        # save model file
        # 保存model文件
        now = time.time()
        if now - last_save_model_time >= 120:
            # 优化：基于综合指标保存最佳模型
            # Optimization: Save best model based on comprehensive metrics
            current_performance = avg_step_reward
            if len(treasures_collected_history) > 0:
                # 综合考虑奖励、宝箱数量和步数
                # Comprehensive consideration of rewards, treasure count, and steps
                normalized_treasures = min(1.0, avg_treasures_collected / 13.0)
                normalized_steps = max(0.0, 1.0 - avg_steps_to_complete / 2500.0)
                completion_bonus = episode_success_rate * 2.0
                
                # 综合得分计算
                # Comprehensive score calculation
                current_performance = avg_step_reward + normalized_treasures * 3.0 + normalized_steps * 2.0 + completion_bonus
            
            if current_performance > best_avg_reward:
                best_avg_reward = current_performance
                logger.info(f"New best model saved with performance: {current_performance:.2f}")
                agent.save_model(id="best_model")
            else:
                agent.save_model()

            last_save_model_time = now

        # 计算训练时间并记录
        # Calculate and record training time
        training_time = (time.time() - training_start_time) / 60  # 分钟
        
        # 优化：详细的日志记录，包括课程阶段和性能指标
        # Optimization: Detailed logging including curriculum phase and performance metrics
        logger.info(
            f"Epoch: {epoch}, Phase: {curriculum_phase}, Reward: {avg_step_reward_str}, "
            f"Success Rate: {episode_success_rate:.2f}, Avg Treasures: {avg_treasures_collected:.1f}, "
            f"Avg Steps: {avg_steps_to_complete:.1f}, Time: {training_time:.1f} min"
        )
        
        # 优化：动态学习率调整
        # Optimization: Dynamic learning rate adjustment
        if epoch > 0 and epoch % 5000 == 0:
            # 基于当前性能调整学习率
            # Adjust learning rate based on current performance
            performance_trend = 0
            if len(recent_rewards) > 30:
                # 计算最近奖励的趋势
                # Calculate trend of recent rewards
                recent_avg = np.mean(recent_rewards[-10:])
                previous_avg = np.mean(recent_rewards[-30:-10])
                performance_trend = recent_avg - previous_avg
            
            if performance_trend > 0.2:
                # 性能显著提升，保持学习率
                # Significant performance improvement, maintain learning rate
                pass
            elif performance_trend > 0:
                # 性能略有提升，小幅降低学习率
                # Slight performance improvement, slightly decrease learning rate
                agent.lr *= 0.98
            else:
                # 性能未提升或下降，较大幅度降低学习率
                # No improvement or decline, decrease learning rate more significantly
                agent.lr *= 0.95
            
            # 更新优化器中的学习率
            # Update learning rate in optimizer
            for param_group in agent.optim.param_groups:
                param_group['lr'] = agent.lr * (param_group['lr'] / agent.lr)  # 保持原有比例
                
            logger.info(f"Learning rate adjusted to {agent.lr:.6f} based on performance trend {performance_trend:.3f}")


def run_episodes(n_episode, env, agent, g_data_truncat, usr_conf, logger):
    episode_rewards = []
    episode_steps = []
    episode_treasures = []
    
    for episode in range(n_episode):
        collector = list()

        # Reset the game and get the initial state
        # 重置游戏, 并获取初始状态
        obs = env.reset(usr_conf=usr_conf)
        env_info = None

        # Disaster recovery
        # 容灾
        if obs is None:
            continue

        # At the start of each game, support loading the latest model file
        # The call will load the latest model from a remote training node
        # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
        agent.load_model(id="latest")

        # Feature processing
        # 特征处理
        obs_data = observation_process(obs)

        done = False
        step = 0
        bump_cnt = 0
        
        # 优化：追踪已收集宝箱以用于启发式探索
        # Optimization: Track collected treasures for heuristic exploration
        collected_treasures = set()
        visited_positions = set()  # 追踪访问过的位置
        repeated_position_count = 0  # 重复位置计数
        last_positions = []  # 最近的几个位置
        
        # 优化：记录起点距离，用于评估探索效率
        # Optimization: Record start distance for exploration efficiency evaluation
        start_pos = None
        if env_info:
            start_pos = (env_info.frame_state.heroes[0].pos.x, env_info.frame_state.heroes[0].pos.z)

        while not done:
            # 优化：智能探索策略
            # Optimization: Intelligent exploration strategy
            
            # Agent performs inference, gets the predicted action for the next frame
            # Agent 进行推理, 获取下一帧的预测动作
            act_data = agent.predict(list_obs_data=[obs_data])[0]

            # Unpack ActData into action
            # ActData 解包成动作
            act = action_process(act_data)

            # 记录当前位置用于检测重复访问
            # Record current position to detect repeated visits
            if env_info:
                current_pos = (env_info.frame_state.heroes[0].pos.x, env_info.frame_state.heroes[0].pos.z)
                visited_positions.add(current_pos)
                last_positions.append(current_pos)
                if len(last_positions) > 10:
                    last_positions.pop(0)
                    
                # 检测在小范围内来回移动的情况
                # Detect back-and-forth movement in a small area
                if len(last_positions) == 10 and len(set(last_positions)) <= 3:
                    repeated_position_count += 1
                    
                    # 如果检测到智能体卡住，使用闪现技能尝试脱困
                    # If agent is detected to be stuck, use flash skill to try to escape
                    if repeated_position_count >= 3:
                        # 强制使用闪现技能
                        # Force use of flash skill
                        act = max(act, 8)  # 激活闪现
                        repeated_position_count = 0

            # Interact with the environment, execute actions, get the next state
            # 与环境交互, 执行动作, 获取下一步的状态
            frame_no, _obs, score, terminated, truncated, _env_info = env.step(act)
            if _obs is None:
                break

            step += 1

            # Feature processing
            # 特征处理
            _obs_data = observation_process(_obs, _env_info)

            # Disaster recovery
            # 容灾
            if truncated and frame_no is None:
                break

            treasures_num = 0

            # Calculate reward
            # 计算 reward
            if env_info is None:
                reward = 0
            else:
                reward, is_bump = reward_shaping(
                    frame_no,
                    score,
                    terminated,
                    truncated,
                    obs,
                    _obs,
                    env_info,
                    _env_info,
                )

                treasure_dists = [pos.grid_distance for pos in _obs.feature.treasure_pos]
                treasures_num = treasure_dists.count(1.0)
                
                # 更新已收集宝箱集合
                # Update collected treasures set
                prev_treasure_dists = [pos.grid_distance for pos in obs.feature.treasure_pos]
                if treasures_num > prev_treasure_dists.count(1.0):
                    # 新收集了宝箱
                    collected_treasures.add(treasures_num)

                # Wall bump behavior statistics
                # 撞墙行为统计
                bump_cnt += is_bump

            # Determine game over, and update the number of victories
            # 判断游戏结束, 并更新胜利次数
            if truncated:
                logger.info(
                    f"Episode {episode} timeout, treasures: {treasures_num - 7}, steps: {step}"
                )
            elif terminated:
                logger.info(
                    f"Episode {episode} reached end, treasures: {treasures_num - 7}, steps: {step}"
                )
            done = terminated or truncated
            
            # 记录episode信息
            # Record episode information
            if done:
                episode_rewards.append(reward)
                episode_steps.append(step)
                episode_treasures.append(treasures_num - 7)

            # Construct game frames to prepare for sample construction
            # 构造游戏帧，为构造样本做准备
            frame = Frame(
                obs=obs_data.feature,
                _obs=_obs_data.feature,
                obs_legal=obs_data.legal_act,
                _obs_legal=_obs_data.legal_act,
                act=act,
                rew=reward,
                done=done,
                ret=reward if done else 0,  # 只在结束时返回累积奖励
            )

            collector.append(frame)

            # If the number of game frames reaches the threshold, the sample is processed and sent to training
            # 如果游戏帧数达到阈值，则进行样本处理，将样本送去训练
            if len(collector) % g_data_truncat == 0:
                collector = sample_process(collector)
                yield collector

            # If the game is over, the sample is processed and sent to training
            # 如果游戏结束，则进行样本处理，将样本送去训练
            if done:
                if len(collector) > 0:
                    # 优化：设置最终奖励以反映全局表现
                    # Optimization: Set final reward to reflect global performance
                    if terminated:
                        # 成功到达终点，计算全局收益
                        # Successfully reached the endpoint, calculate global performance
                        treasure_ratio = (treasures_num - 7) / 13.0  # 宝箱收集比例
                        step_efficiency = max(0, 1.0 - step / int(usr_conf["diy"]["max_step"]))  # 步数效率
                        global_performance = treasure_ratio * 100 + step_efficiency * 50  # 全局性能评分
                        
                        # 设置更好的回报作为评估反馈
                        # Set better return as evaluation feedback
                        for i in range(max(0, len(collector) - 1), len(collector)):
                            collector[i].ret = global_performance
                    
                    collector = sample_process(collector)
                    yield collector
                break

            # Status update
            # 状态更新
            obs_data = _obs_data
            obs = _obs
            env_info = _env_info
