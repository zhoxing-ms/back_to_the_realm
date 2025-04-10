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

# 全局参数：检查是否仅有两个宝箱
CHECK_TWO_TREASURES = True

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
    remaining_treasure_dists = [d for d in treasure_dists if d != 1.0]

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
    prev_remaining_treasure_dists = [d for d in prev_treasure_dists if d != 1.0]

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
    has2Treasures = len(remaining_treasure_dists) == 2 and len(prev_remaining_treasure_dists) == 2
    total_collected_treasures = treasure_dists.count(1.0)

    # 获取动作信息，以判断是否使用了闪现
    prev_status = env_info.frame_state.heroes[0].talent.status
    curr_status = _env_info.frame_state.heroes[0].talent.status
    is_talent_used = ( prev_status == 1 and curr_status == 0 )

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
    # Reward 2.1 Reward for getting closer to the treasure chest
    # 奖励2.1 向宝箱靠近的奖励(优先考虑不顺路宝箱中最近的，其次考虑顺路宝箱中最近的)
    if treasure_dists.count(1.0) < len(treasure_dists):
        if has2Treasures and CHECK_TWO_TREASURES:
            endCloser = end_dist < prev_end_dist
            prev_treasure_0 = prev_remaining_treasure_dists[0]
            prev_treasure_1 = prev_remaining_treasure_dists[1]
            treasure_0 = remaining_treasure_dists[0]
            treasure_1 = remaining_treasure_dists[1]
            treasure_0_closer = treasure_0 < prev_treasure_0
            treasure_1_closer = treasure_1 < prev_treasure_1
            if treasure_0_closer and treasure_1_closer:
                reward_treasure_dist += 2
            elif (treasure_0_closer or treasure_1_closer):
                if endCloser:
                    reward_treasure_dist -= 20
                else:
                    reward_treasure_dist += 2
            else:
                reward_treasure_dist -= 2
        else:
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

    # 为闪现制定奖励/惩罚策略
    if is_talent_used:

        if is_bump:
            # 撞墙闪现的惩罚
            reward_flicker -= 200
        else:
            # 正常闪现的奖励
            if is_treasures_remain:
                # 如果还有宝箱，奖励靠近未获取宝箱的闪现
                min_remaining_treasure_dist = min(remaining_treasure_dists)
                prev_min_remaining_treasure_dist = min(prev_remaining_treasure_dists) if prev_remaining_treasure_dists else 1.0 # 不加防止空数组的逻辑判定会报错，可能和视野有关
                if min_remaining_treasure_dist < prev_min_remaining_treasure_dist:
                    reward_flicker += 30 * (prev_min_remaining_treasure_dist - min_remaining_treasure_dist)
            else:
                # 如果宝箱已经收集完，奖励靠近终点的闪现
                if end_dist < prev_end_dist:
                    reward_flicker += 30 * (prev_end_dist - end_dist)


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
    def calculate_area_visit_penalty(memory_map, area_size=7):
        """
        通过检查7*7区域的四个边界来判断轨迹数量，累加边界点的访问次数后除2为轨迹数
        Args:
            memory_map: 探索记录地图（一维数组，实际表示51x51的地图）
            area_size: 检查区域的大小，默认7表示7x7的区域
        Returns:
            penalty: 区域重复探索的惩罚值
        """
        map_size = 51
        center = len(memory_map) // 2
        half_size = area_size // 2
        
        # 将7*7区域转换为二维数组
        area_map = np.zeros((area_size, area_size))
        
        # 提取7*7区域的访问记录
        for row in range(-half_size, half_size + 1):
            for col in range(-half_size, half_size + 1):
                idx = center + col + row * map_size
                if 0 <= idx < len(memory_map):
                    area_map[row + half_size][col + half_size] = memory_map[idx]
        
        # 检查四个边界的访问次数总和
        boundary_visits = 0  # 记录边界上的总访问次数
        
        # 遍历边界
        for i in range(area_size):
            # 检查左边界列
            boundary_visits += area_map[i][0]
            # 检查右边界列
            boundary_visits += area_map[i][area_size-1]
            
            # 检查上下边界行（排除已经计算过的角点）
            if i != 0 and i != area_size-1:  # 排除与左右边界重复计算的点
                # 检查上边界
                boundary_visits += area_map[0][i]
                # 检查下边界
                boundary_visits += area_map[area_size-1][i]
        
        # 计算轨迹数量和惩罚
        penalty = 0
        
        # 计算实际轨迹数（边界访问总次数除以2）
        num_paths = boundary_visits / 2
        
        # 如果轨迹数超过2条，进行惩罚
        if num_paths > 2:
            penalty += (num_paths - 2) * 30  # 每多一条轨迹增加30的惩罚
        
        # 设置惩罚上限
        final_penalty = min(penalty, 500)
        
        return final_penalty

    memory_map = obs.feature.memory_map
    reward_memory = calculate_area_visit_penalty(memory_map, area_size=7)
    
    if not is_treasures_remain:
        # 剩余宝箱时惩罚
        reward_memory = calculate_area_visit_penalty(memory_map, area_size=7)

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
        "reward_memory": "-0.005",          # 复探索惩罚
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
