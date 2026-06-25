# From Developing to Exploiting Quoridor Agents

这是 Group 18 的 CS181 人工智能课程 Project。项目主题来自 proposal：**From Developing to Exploiting Quoridor Agents: Minimal Adversarial Policies Against Search and Learning Baselines**。

项目不是单纯堆叠更强的 Quoridor agent，而是研究一个更具体的问题：

> 如何用较简单、可解释的 adversarial policy，稳定利用 Greedy BFS、depth-limited Minimax、finite-budget MCTS，以及后续 deterministic RL / NN policy 的盲点？

当前仓库已经把这个研究问题落到了一个可运行的 Python 环境中：

- Quoridor 核心规则引擎
- Tkinter 图形界面
- 像素化 3D Web 图形界面，支持 Human / Agent 任意组合对弈
- RandomAgent 示例
- Greedy BFS / Minimax / MCTS baseline agents
- PathLure / DepthTrap / RolloutPoison adversarial agents
- Tournament / Elo evaluation framework
- 固定离散动作空间的训练接口
- 单元测试

## Proposal 对齐和完成状态

proposal 的核心分解如下：

| Proposal 目标 | 当前状态 | 对应代码 |
| --- | --- | --- |
| 标准 Quoridor 规则、移动、跳跃、放墙和通路合法性 | 已实现 | `quoridor/core/` |
| Greedy BFS baseline | 已实现 | `quoridor/agents/greedy_bfs.py` |
| Minimax + Alpha-Beta baseline | 已实现 | `quoridor/agents/minimax.py` |
| finite-budget UCT MCTS baseline | 已实现 | `quoridor/agents/mcts.py` |
| Path-Lure attack：利用 Greedy BFS 只看短路长度、忽视 path diversity | 已实现 | `PathLureAgent` |
| Depth-Trap attack：利用 depth-limited Minimax 的短视 horizon | 已实现 | `DepthTrapAgent` |
| Rollout-Poison attack：利用有限 rollout 下 response 分数接近、难区分 | 已实现 | `RolloutPoisonAgent` |
| Elo tournament、ablation、trap proxy 指标 | 已实现轻量框架 | `quoridor/evaluation/`, `experiments/` |
| 用于展示和调试 Human / Agent / Agent-vs-Agent 的可视化界面 | 已实现 | `visual_web.py`, `quoridor/web/` |
| 固定动作空间训练接口，便于后续接 RL | 已实现接口 | `quoridor/training/discrete_env.py` |
| Argmax RL victim exploitation | 未实现实际训练/查询 victim，仅保留训练接口 |
| Neural-pruned MCTS / PUCT / NN value-prior | 未实现，仍是 proposal appendix / future work |
| Approximate Q-Learning / Deep Q network victim | 未实现，仍是 proposal appendix / future work |
| 1000+ rounds 的正式实验结果表 | 未提交正式结果；当前脚本支持跑实验，仓库内只保留轻量 smoke 输出在 `tmp/` |

因此，当前项目主线已经覆盖 proposal 的传统 search baseline 和 minimal adversarial policy 部分；神经网络/RL 部分还没有落地为训练代码或可复现实验。

## 运行环境

核心引擎、训练接口、实验脚本和 Tkinter UI 只需要 Python 标准库，不需要额外安装依赖。

Web UI 的 Python server 也只依赖标准库；浏览器前端会从 CDN 加载 Three.js，并使用仓库内的本地 pawn OBJ 模型。如果完全离线，核心引擎和 Tkinter UI 仍可使用，但 Web 3D 棋盘需要提前准备 Three.js 资源。

推荐 Python 版本：Python 3.11+

## 项目结构

```text
AI_Project/
  quoridor/
    core/
      actions.py          # MoveAction / WallAction
      state.py            # 不可变游戏状态 QuoridorState
      rules.py            # 合法动作、跳跃、放墙、通路检查、胜负判断
      env.py              # QuoridorEnv 环境封装
      ascii_board.py      # ASCII 棋盘渲染，用于调试
    agents/
      base.py             # Agent 接口示例
      random_agent.py     # RandomAgent 示例
      greedy_bfs.py       # 最短路贪心 baseline
      minimax.py          # Alpha-beta minimax baseline
      mcts.py             # UCT MCTS baseline
      adversarial.py      # PathLure / DepthTrap / RolloutPoison adversarial agents
      heuristics.py       # 搜索 agent 共享启发式
    evaluation/
      metrics.py          # 单局对战和结果记录
      tournament.py       # Round-robin tournament
      elo.py              # Elo 更新工具
    training/
      discrete_env.py     # 训练用固定动作空间接口
    ui/
      tkinter_app.py      # Tkinter 可视化界面
    web/
      server.py           # 标准库 Web UI server
      static/             # Web 棋盘、控制台、像素后处理和 3D pawn 模型
        models/pawn/      # Web UI 使用的棋子 OBJ 资源
  experiments/
    run_tournament.py     # 小规模 tournament 命令行入口
  tests/
    test_rules.py
    test_random_agent.py
    test_search_agents.py
    test_evaluation.py
    test_training_env.py
  visual.py               # 启动图形界面
  visual_web.py           # 启动 Web 图形界面
  train_interface_demo.py # 训练接口示例
```

## Web 图形界面

推荐使用 Web UI 做调试和报告展示：

```powershell
python visual_web.py
```

默认地址：

```text
http://127.0.0.1:8765
```

Web UI 支持：

- Human vs Human
- PVE：Human vs Random / Greedy BFS / Minimax / MCTS / PathLure / DepthTrap / RolloutPoison
- Agent vs Agent：任意两个内置 agent 自动对弈
- `Step` 单步执行当前 agent
- `Play/Pause` 自动播放 agent 回合
- `Reset` 重开
- `Move / Horizontal wall / Vertical wall` 人类操作模式
- 合法走法提示、合法墙位 ghost 预览和悬停高亮
- 最短路径、path diversity、剩余墙数、回合数和 action log
- 像素化后处理、左右玩家状态面板、wall rack 和 3D pawn 模型

Web UI 的后端只依赖 Python 标准库；前端通过 CDN 加载 Three.js / OrbitControls / OBJLoader / postprocessing 模块。当前没有使用 GSAP。

## Tkinter 图形界面

运行：

```powershell
python visual.py
```

界面支持：

- Human vs Human
- Human vs Random
- Random vs Human
- Random vs Random

顶部可以分别设置 `Player 0` 和 `Player 1` 是 `Human` 还是 `Random`。

操作方式：

- 选择 `Move` 后，点击目标格移动棋子。
- 选择 `Horizontal wall` 后，点击横墙位置放横墙。
- 选择 `Vertical wall` 后，点击竖墙位置放竖墙。
- 非法动作会被引擎拒绝。
- `Reset` 可以重开一局。

## 核心引擎用法

```python
from quoridor import MoveAction, QuoridorEnv, WallAction

env = QuoridorEnv()

print(env.state)
print(env.legal_actions())

env.step(MoveAction((7, 4)))
env.step(WallAction("H", 1, 4))
```

`env.step(action)` 会返回 `StepResult`：

```python
result.state   # 新状态
result.reward  # (player0_reward, player1_reward)
result.done    # 是否结束
result.winner  # 赢家，未结束时为 None
```

## Agent 接口

其他同学写智能体时，只需要实现类似接口：

```python
class MyAgent:
    def choose_action(self, state, legal_actions):
        return legal_actions[0]
```

示例 RandomAgent：

```python
from quoridor import QuoridorEnv
from quoridor.agents import RandomAgent

env = QuoridorEnv()
agent = RandomAgent(seed=0)

action = agent.choose_action(env.state, env.legal_actions())
env.step(action)
```

当前已经内置几个 Project baseline：

```python
from quoridor.agents import DepthTrapAgent, GreedyBFSAgent, MCTSAgent, MinimaxAgent, PathLureAgent, RolloutPoisonAgent

greedy = GreedyBFSAgent(seed=0)
minimax = MinimaxAgent(depth=2, action_limit=24, wall_limit=16)
mcts = MCTSAgent(iterations=100, rollout_depth=24)
path_lure = PathLureAgent(seed=0)
depth_trap = DepthTrapAgent(seed=0)
rollout_poison = RolloutPoisonAgent(seed=0)
```

这些 agent 都只依赖标准 `choose_action(state, legal_actions)` 接口。搜索类 agent 默认会先用启发式筛选候选墙位，再用完整规则引擎验证合法性，避免在实验中反复枚举全棋盘墙位导致运行过慢。

- `PathLureAgent`：针对 Greedy BFS，奖励让对手 path diversity 下降的动作。
- `DepthTrapAgent`：针对 depth-limited Minimax，奖励对手浅层响应后我方仍有强 follow-up 的动作。
- `RolloutPoisonAgent`：针对有限预算 MCTS，奖励对手浅 rollout 下响应分数接近、难以区分的局面。

这些 adversarial agents 是 proposal 主线的最小可运行版本。它们不是通用最强 agent，而是面向特定 victim blind spot 的 exploit policy。

## 对战评估

运行一个轻量 smoke tournament：

```powershell
python experiments\run_tournament.py --preset smoke --games-per-pair 1 --max-turns 20 --output tmp\tournament_smoke.csv
```

运行包含当前所有已实现 agent 的轻量完整 preset：

```powershell
python experiments\run_tournament.py --preset full --games-per-pair 2 --max-turns 150 --output experiments\results\tournament_games.csv
```

运行 adversarial preset：

```powershell
python experiments\run_tournament.py --preset adversarial --games-per-pair 1 --max-turns 80 --output experiments\results\tournament_adversarial.csv
```

运行更重的 research preset：

```powershell
python experiments\run_tournament.py --preset research --games-per-pair 2 --max-turns 150 --output experiments\results\tournament_research.csv
```

运行 PathLure 的 trap-weight ablation：

```powershell
python experiments\run_ablation.py --weights 0,4,8 --games 4 --max-turns 80 --output experiments\results\path_lure_ablation.csv
```

代码接口：

```python
from quoridor.agents import GreedyBFSAgent, RandomAgent
from quoridor.evaluation import AgentSpec, run_round_robin

result = run_round_robin(
    [
        AgentSpec("random", lambda: RandomAgent(seed=0)),
        AgentSpec("greedy", lambda: GreedyBFSAgent(seed=1)),
    ],
    games_per_pair=2,
    max_turns=100,
)

print(result.standings())
result.write_games_csv("tmp/games.csv")
```

CSV 会记录胜负、回合数、剩余墙数、起止最短路、最终/最小 path diversity、移动/放墙次数，以及 `trap_events`。`trap_events` 是一个实验 proxy：当行动方让对手的最短路选择数降到 `1` 或以下，同时对手最短路至少比开局长 `1` 步时计数。

## 训练接口

训练接口在：

```text
quoridor/training/discrete_env.py
```

它把所有动作编码成固定大小的离散动作空间，动作总数是 `209`：

```text
0 - 80      移动棋子到 9x9 某个格子
81 - 144    放横墙 H，8x8 个位置
145 - 208   放竖墙 V，8x8 个位置
```

基本用法：

```python
from quoridor.training import DiscreteQuoridorEnv

env = DiscreteQuoridorEnv()
obs = env.reset()

legal_mask = obs["legal_action_mask"]
legal_ids = [i for i, ok in enumerate(legal_mask) if ok]

action_id = legal_ids[0]
result = env.step(action_id)

next_obs = result.observation
reward = result.reward
done = result.done
```

观测 `obs` 是一个字典：

```python
{
    "current_player": 0,
    "pawn_planes": ...,        # 2 个 9x9 平面
    "horizontal_walls": ...,   # 8x8 平面
    "vertical_walls": ...,     # 8x8 平面
    "remaining_walls": [10, 10],
    "legal_action_mask": [...],
    "done": False,
    "winner": None,
}
```

如果模型需要一维输入：

```python
flat = env.flat_observation()
```

训练接口 demo：

```powershell
python train_interface_demo.py
```

## 非法动作处理

默认情况下，训练环境遇到非法动作会直接抛出异常：

```python
env = DiscreteQuoridorEnv()
```

如果希望非法动作返回惩罚而不是报错：

```python
env = DiscreteQuoridorEnv(invalid_action_penalty=-1)
```

此时非法动作不会推进回合，会给当前玩家 `-1` 奖励。

## 测试

运行全部测试：

```powershell
python -m unittest discover -s tests -v
```

当前测试覆盖：

- 初始状态
- 普通移动
- 跳跃和侧跳
- 墙阻挡
- 墙重叠/交叉检测
- 不能完全封死路径
- RandomAgent
- Greedy BFS / Minimax / MCTS / PathLure / DepthTrap / RolloutPoison 返回合法动作
- 单局对战、round-robin tournament 和 Elo 更新
- 训练接口动作编码和合法动作 mask
- Web session payload、agent 配置和 agent step

## 交接说明

核心规则都在 `quoridor/core/`。一般情况下，后续同学不需要改规则引擎，只需要：

- 写自己的 agent：参考 `quoridor/agents/random_agent.py`
- 写搜索 baseline：参考 `quoridor/agents/greedy_bfs.py`、`quoridor/agents/minimax.py`、`quoridor/agents/mcts.py`
- 写 adversarial policy：参考 `quoridor/agents/adversarial.py`
- 跑实验：参考 `experiments/run_tournament.py`
- 接训练代码：参考 `quoridor/training/discrete_env.py`
- 调试 Web 界面：运行 `python visual_web.py`
- 调试 Tkinter 界面：运行 `python visual.py`
- 验证改动：运行 `python -m unittest discover -s tests -v`

如果后续要继续完成 proposal 的 NN/RL 部分，建议先补：

- 一个实际可训练/可加载的 deterministic Q victim，并暴露 `choose_action(state, legal_actions)`；
- 一个 neural-pruned MCTS / PUCT baseline，而不是只在 appendix 中描述；
- 固定 seeds、games-per-pair、max-turns 的正式 tournament 输出，并把结果 CSV 或汇总表纳入报告材料。
