#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :back_to_the_realm
@File    :definition.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""


from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np
from kaiwu_agent.back_to_the_realm.diy.feature_process import (
    one_hot_encoding,
    read_relative_position,
    bump,
)


# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    move_dir=None,
    use_talent=None,
)


SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    obs_legal=None,
    _obs_legal=None,
    act=None,
    rew=None,
    ret=None,
    done=None,
)


def reward_shaping(frame_no, score, terminated, truncated, obs, _obs, env_info, _env_info):
    reward = 0

    # Get the current position coordinates of the agent
    # 获取当前智能体的位置坐标
    pos = _env_info.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = pos.x, pos.z

    # Get the grid-based distance of the current agent's position relative to the end point, buff, and treasure chest
    # 获取当前智能体的位置相对于终点, buff, 宝箱的栅格化距离
    end_dist = _obs.feature.end_pos.grid_distance
    buff_dist = _obs.feature.buff_pos.grid_distance
    treasure_dists = [pos.grid_distance for pos in _obs.feature.treasure_pos]

    # Get the agent's position from the previous frame
    # 获取智能体上一帧的位置
    prev_pos = env_info.frame_state.heroes[0].pos
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z

    # Get the grid-based distance of the agent's position from the previous
    # frame relative to the end point, buff, and treasure chest
    # 获取智能体上一帧相对于终点，buff, 宝箱的栅格化距离
    prev_end_dist = obs.feature.end_pos.grid_distance
    prev_buff_dist = obs.feature.buff_pos.grid_distance
    prev_treasure_dists = [pos.grid_distance for pos in obs.feature.treasure_pos]

    # Get the status of the buff
    # 获取buff的状态
    buff_availability = 0
    for organ in env_info.frame_state.organs:
        if organ.sub_type == 2:
            buff_availability = organ.status

    # Get the acceleration status of the agent
    # 获取智能体的加速状态
    prev_speed_up = env_info.frame_state.heroes[0].speed_up
    speed_up = _env_info.frame_state.heroes[0].speed_up

    # Are there any remaining treasure chests
    # 是否有剩余宝箱
    is_treasures_remain = True if treasure_dists.count(1.0) < len(treasure_dists) else False
    total_collected_treasures = treasure_dists.count(1.0)

    # 获取动作信息，以判断是否使用了闪现
    is_talent_used = False
    move_dir = 0
    use_talent = 0

    # 检查当前帧与前一帧之间的位置变化，大幅度变化可能表示使用了闪现
    if env_info and _env_info:
        prev_pos = env_info.frame_state.heroes[0].pos
        curr_pos = _env_info.frame_state.heroes[0].pos
        distance = np.sqrt((curr_pos.x - prev_pos.x) ** 2 + (curr_pos.z - prev_pos.z) ** 2)

        # 如果距离大于正常移动可达的距离，判定为使用了闪现
        is_talent_used = distance > 4000  # 阈值可以根据实际情况调整

        # 计算移动方向
        if distance > 0:
            angle = np.arctan2(curr_pos.z - prev_pos.z, curr_pos.x - prev_pos.x)
            move_dir = int((angle + np.pi) / (np.pi / 4)) % 8

    """
    Reward 1. Reward related to the end point
    奖励1. 与终点相关的奖励
    """
    reward_end_dist = 0
    # Reward 1.1 Reward for getting closer to the end point
    # 奖励1.1 向终点靠近的奖励

    # Boundary handling: At the first frame, prev_end_dist is initialized to 1,
    # and no reward is calculated at this time
    # 边界处理: 第一帧时prev_end_dist初始化为1，此时不计算奖励
    if prev_end_dist != 1 and not is_treasures_remain:
        reward_end_dist += 2 if end_dist < prev_end_dist else -2

    # Reward 1.2 Reward for winning
    # 奖励1.2 获胜的奖励
    reward_win = 0
    if terminated:
        if not is_treasures_remain:
            reward_win += 2
        else:
            reward_win -= 1

    """
    Reward 2. Rewards related to the treasure chest
    奖励2. 与宝箱相关的奖励
    """
    reward_treasure_dist = 0
    # Reward 2.1 Reward for getting closer to the treasure chest (only consider the nearest one)
    # 奖励2.1 向宝箱靠近的奖励(只考虑最近的那个宝箱)
    if treasure_dists.count(1.0) < len(treasure_dists):
        prev_min_dist, min_dist = min(prev_treasure_dists), min(treasure_dists)
        if prev_treasure_dists.index(prev_min_dist) == treasure_dists.index(min_dist):
            reward_treasure_dist += 2 if min_dist < prev_min_dist else -2

    # Reward 2.2 Reward for getting the treasure chest
    # 奖励2.2 获得宝箱的奖励
    reward_treasure = 0
    if prev_treasure_dists.count(1.0) < treasure_dists.count(1.0):
        reward_treasure = 2

    """
    Reward 3. Rewards related to the buff
    奖励3. 与buff相关的奖励
    """
    # Reward 3.1 Reward for getting closer to the buff
    # 奖励3.1 靠近buff的奖励
    reward_buff_dist = 0
    if prev_buff_dist != 1 and buff_dist != 1 and buff_availability:
        reward_buff_dist = int(256 * (prev_buff_dist - buff_dist))

    # Reward 3.2 Reward for getting the buff
    # 奖励3.2 获得buff的奖励
    reward_buff = 0
    if speed_up and not prev_speed_up:
        reward_buff += 50

    """
    Reward 4. Rewards related to the flicker
    奖励4. 与闪现相关的奖励
    """
    reward_flicker = 0
    # Reward 4.1 Penalty for flickering into the wall
    # 奖励4.1 撞墙闪现的惩罚
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)
    if is_talent_used and is_bump:
        # 撞墙闪现的惩罚应该比普通撞墙大，因为浪费了一个重要的技能
        reward_flicker -= 50

    # Reward 4.2 Reward for normal flickering
    # 奖励4.2 正常闪现的奖励
    elif is_talent_used and not is_bump:
        # 通过观察目标位置是否更靠近目标来判断闪现质量
        if is_treasures_remain:
            # 如果还有宝箱，奖励靠近宝箱的闪现
            min_treasure_dist = min(treasure_dists)
            prev_min_treasure_dist = min(prev_treasure_dists)
            if min_treasure_dist < prev_min_treasure_dist:
                reward_flicker += 30 * (prev_min_treasure_dist - min_treasure_dist)
        else:
            # 如果宝箱已经收集完，奖励靠近终点的闪现
            if end_dist < prev_end_dist:
                reward_flicker += 30 * (prev_end_dist - end_dist)

    # Reward 4.3 Reward for super flickering
    # 奖励4.3 超级闪现的奖励
    # 奖励4.3 超级闪现的奖励（穿墙或跨越大障碍）
    # 超级闪现可以定义为能够跨越障碍物的闪现
    # 可以通过检查闪现前后路径距离与直线距离的差异来判断
    if is_talent_used and not is_bump:
        # 假设闪现前后的网格路径距离变化显著大于实际直线距离，说明闪现穿越了障碍物
        if is_treasures_remain:
            # 当还有宝箱时，计算当前与最近宝箱的路径差异
            min_treasure_idx = treasure_dists.index(min(treasure_dists))
            prev_min_treasure_idx = prev_treasure_dists.index(min(prev_treasure_dists))
            
            if min_treasure_idx == prev_min_treasure_idx:
                # 计算路径距离与直线距离的比值，比值大说明有障碍
                path_diff = 256 * (prev_treasure_dists[min_treasure_idx] - treasure_dists[min_treasure_idx])
                if path_diff > 10:  # 阈值可调
                    reward_flicker += 40  # 额外奖励穿墙闪现
        else:
            # 已收集所有宝箱时，计算与终点的路径差异
            path_diff = 256 * (prev_end_dist - end_dist)
            if path_diff > 10:  # 阈值可调
                reward_flicker += 40  # 额外奖励穿墙闪现

    """
    Reward 5. Rewards for quick clearance
    奖励5. 关于快速通关的奖励
    """
    reward_step = 1
    step_penalty_scale = 0.001  # 基础步数惩罚尺度
    
    # 基于进度调整步数惩罚，前期较小，后期较大
    # Adjust step penalty based on progress, smaller in early stages, larger in later stages
    if frame_no <= 1000:
        # 前期小惩罚，允许探索
        reward_step = step_penalty_scale * 0.5
    elif frame_no <= 2000:
        reward_step = step_penalty_scale * 1.0
    else:
        # 后期大惩罚，促进高效
        reward_step = step_penalty_scale * (1.0 + total_collected_treasures / len(treasure_dists))

    # Reward 5.1 Reward/penalty for approaching endpoint after collecting all treasures
    # 奖励5.1 收集完所有宝箱后靠近终点的奖励/惩罚
    reward_treasure_all_collected_dist = 0
    if not is_treasures_remain:
        # 如果所有宝箱都已收集，对不靠近终点的行为进行惩罚，使用终点距离作为塑形奖励引导智能体前往终点
        # 靠近终点给奖励，远离终点给惩罚
        end_dist_change = prev_end_dist - end_dist
        reward_treasure_all_collected_dist = end_dist_change * 30

    # Reward 5.2 Penalty for repeated exploration
    # 奖励5.2 重复探索的惩罚
    reward_memory = 0
    memory_map = obs.feature.memory_map
    reward_memory = memory_map[len(memory_map) // 2]

    # Reward 5.3 Penalty for bumping into the wall
    # 奖励5.3 撞墙的惩罚
    reward_bump = 0
    # Determine whether it bumps into the wall
    # 判断是否撞墙
    if is_bump:
        # Give a relatively large penalty for bumping into the wall,
        # so that the agent can learn not to bump into the wall as soon as possible
        # 对撞墙给予一个比较大的惩罚，以便agent能够尽快学会不撞墙
        reward_bump = 200

    """
    Concatenation of rewards: Here are 12 rewards provided,
    students can concatenate as needed, and can also add new rewards themselves
    奖励的拼接: 这里提供了11个奖励, 同学们按需自行拼接, 也可以自行添加新的奖励
    """
    REWARD_CONFIG = {
        "reward_end_dist": "0.1",           # 接近终点的奖励
        "reward_win": "0.2",                # 成功达到终点的奖励
        "reward_buff_dist": "0.1",          # 增加接近buff的奖励
        "reward_buff": "0.15",              # 增加获取buff的奖励
        "reward_treasure_dist": "0.1",     # 接近宝箱的奖励
        "reward_treasure": "0.15",          # 获取宝箱的奖励
        "reward_flicker": "0.1",            # 增加闪现的奖励
        "reward_treasure_all_collected_dist": "0.1", # 收集完宝箱后引导终点的奖励
        "reward_step": "-0.001",           # 步数惩罚以鼓励更快完成
        "reward_bump": "-0.005",            # 撞墙惩罚
        "reward_memory": "-0.015",          # 增大复探索惩罚
    }

    reward = [
        reward_end_dist * float(REWARD_CONFIG["reward_end_dist"]),
        reward_win * float(REWARD_CONFIG["reward_win"]),
        reward_buff_dist * float(REWARD_CONFIG["reward_buff_dist"]),
        reward_buff * float(REWARD_CONFIG["reward_buff"]),
        reward_treasure_dist * float(REWARD_CONFIG["reward_treasure_dist"]),
        reward_treasure * float(REWARD_CONFIG["reward_treasure"]),
        reward_flicker * float(REWARD_CONFIG["reward_flicker"]),
        reward_step * float(REWARD_CONFIG["reward_step"]),
        reward_bump * float(REWARD_CONFIG["reward_bump"]),
        reward_memory * float(REWARD_CONFIG["reward_memory"]),
        reward_treasure_all_collected_dist * float(REWARD_CONFIG["reward_treasure_all_collected_dist"]),
    ]

    return sum(reward), is_bump


@attached
def observation_process(raw_obs, env_info=None):
    """
    This function is an important feature processing function, mainly responsible for:
        - Parsing information in the raw data
        - Parsing preprocessed feature data
        - Processing the features and returning the processed feature vector
        - Concatenation of features
        - Annotation of legal actions
    Function inputs:
        - raw_obs: Preprocessed feature data
        - env_info: Environment information returned by the game
    Function outputs:
        - observation: Feature vector
        - legal_action: Annotation of legal actions

    该函数是特征处理的重要函数, 主要负责：
        - 解析原始数据里的信息
        - 解析预处理后的特征数据
        - 对特征进行处理, 并返回处理后的特征向量
        - 特征的拼接
        - 合法动作的标注
    函数的输入：
        - raw_obs: 预处理后的特征数据
        - env_info: 游戏返回的环境信息
    函数的输出：
        - observation: 特征向量
        - legal_action: 合法动作的标注
    """
    feature, legal_act = [], []

    # Unpack the preprocessed feature data according to the protocol
    # 对预处理后的特征数据按照协议进行解包
    norm_pos = raw_obs.feature.norm_pos
    grid_pos = raw_obs.feature.grid_pos
    start_pos = raw_obs.feature.start_pos
    end_pos = raw_obs.feature.end_pos
    buff_pos = raw_obs.feature.buff_pos
    treasure_poss = raw_obs.feature.treasure_pos
    obstacle_map = list(raw_obs.feature.obstacle_map)
    memory_map = list(raw_obs.feature.memory_map)
    treasure_map = list(raw_obs.feature.treasure_map)
    end_map = list(raw_obs.feature.end_map)

    # Feature processing 1: One-hot encoding of the current position
    # 特征处理1：当前位置的one-hot编码
    one_hot_pos = one_hot_encoding(grid_pos)

    # Feature processing 2: Normalized position
    # 特征处理2：归一化位置
    norm_pos = [norm_pos.x, norm_pos.z]

    # Feature processing 3: Information about the current position relative to the end point
    # 特征处理3：当前位置相对终点点位的信息
    end_pos_features = read_relative_position(end_pos)

    # Feature processing 4: Information about the current position relative to the treasure position
    # 特征处理4: 当前位置相对宝箱位置的信息
    treasure_poss_features = []
    for treasure_pos in treasure_poss:
        treasure_poss_features = treasure_poss_features + list(read_relative_position(treasure_pos))

    # Feature processing 5: Whether the buff is collectable
    # 特征处理5：buff是否可收集
    buff_availability = 0
    if env_info:
        for organ in env_info.frame_state.organs:
            if organ.sub_type == 2:
                buff_availability = organ.status

    # Feature processing 6: Whether the flash skill can be used
    # 特征处理6：闪现技能是否可使用
    talent_availability = 0
    if env_info:
        talent_availability = env_info.frame_state.heroes[0].talent.status

    # Feature processing 7: Next treasure chest to find
    # 特征处理7：下一个需要寻找的宝箱
    treasure_dists = [pos.grid_distance for pos in treasure_poss]
    if treasure_dists.count(1.0) < len(treasure_dists):
        end_treasures_id = np.argmin(treasure_dists)
        end_pos_features = read_relative_position(treasure_poss[end_treasures_id])

    # Feature concatenation:
    # Concatenate all necessary features as vector features (2 + 128*2 + 9  + 9*15 + 2 + 4*51*51 = 10808)
    # 特征拼接：将所有需要的特征进行拼接作为向量特征 (2 + 128*2 + 9  + 9*15 + 2 + 4*51*51 = 10808)
    feature_vec = (
        norm_pos + one_hot_pos + end_pos_features + treasure_poss_features + [buff_availability, talent_availability]
    )
    feature_map = obstacle_map + end_map + treasure_map + memory_map
    # Legal actions
    # 合法动作
    legal_act = list(raw_obs.legal_act)

    return ObsData(feature=feature_vec + feature_map, legal_act=legal_act)


@attached
def action_process(act_data):
    result = act_data.move_dir
    result += act_data.use_talent * 8
    return result


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


# SampleData <----> NumpyData
@attached
def SampleData2NumpyData(g_data):
    return np.hstack(
        (
            np.array(g_data.obs, dtype=np.float32),
            np.array(g_data._obs, dtype=np.float32),
            np.array(g_data.obs_legal, dtype=np.float32),
            np.array(g_data._obs_legal, dtype=np.float32),
            np.array(g_data.act, dtype=np.float32),
            np.array(g_data.rew, dtype=np.float32),
            np.array(g_data.ret, dtype=np.float32),
            np.array(g_data.done, dtype=np.float32),
        )
    )


@attached
def NumpyData2SampleData(s_data):
    return SampleData(
        # Refer to the DESC_OBS_SPLIT configuration in config.py for dimension reference
        # 维度参考config.py 中的 DESC_OBS_SPLIT配置
        obs=s_data[:10808],
        _obs=s_data[10808:21616],
        obs_legal=s_data[-8:-6],
        _obs_legal=s_data[-6:-4],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
