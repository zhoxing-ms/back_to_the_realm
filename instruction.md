请帮我优化这个训练智能体的代码，让其在对地图不断的探索中学习移动策略，减少碰撞障碍物，以最少的步数从起点走到终点，并尽可能多的收集宝箱。

你在优化改进代码时需要格外注意的地方有:
1. 你需要从documents目录下面的简介、环境介绍、代码包介绍、数据协议对整个项目的上下文有更全面深入的理解
2. 目前智能体代码最主要的问题在于探索的路径不是最优解，容易重复探索导致步数增加。而且智能体为了选择靠近最近的宝箱获得立即奖励选择了局部最优解，但是为此忽略了全局最优解，导致重复走了非最优的路线，多出了很多不必要的步骤。你需要考虑优化智能体的探索策略，让智能体在探索时更看重长期全局的回报，能够选择全局的最优的路径，以最少的步数从起点走到终点，并尽可能的收集完全部的宝箱。
3. 每一处你优化的代码和参数都需要加上注释，以此详细的解释你这么优化的原因，要让别人一看注释就很清晰的理解你的思路（比如计算n步回报，结合当前和未来的奖励）
4. 在documents目录下有前人优化经验，你可以参考前人的经验，但是不要完全按照前人的经验来，因为前人的经验不一定适合当前的场景
5. 在优化改进代码时，你需要注意代码的上下文，要用全局的视角来优化，不要破坏代码的完整性、功能性和可读性
