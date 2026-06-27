# Quoridor Project Next Steps Report

更新时间：2026-06-27

这份报告写给还不熟悉代码库的同学。目标是让大家知道项目现在做到哪一步、接下来要跑什么实验、结果文件应该放在哪里，以及论文该怎么收敛。

## 1. 项目一句话说明

我们的课程项目不是单纯做一个最强 Quoridor AI，而是研究：

> 能不能用比较简单、可解释的 adversarial policy，稳定利用 Greedy BFS、Minimax、MCTS、RL/NN baseline 的弱点？

所以论文主线应该一直围绕“开发 baseline，然后利用它们的盲点”。

## 2. 现在已经完成什么

代码主体已经基本完成：

- Quoridor 规则引擎：棋子移动、跳子、侧跳、墙体合法性、不能封死路径、胜负判断。
- Baseline agents：Greedy BFS、Minimax、MCTS、Q-learning、Approx-Q、Deep-Q、PUCT。
- Trap agents：PathLure、DepthTrap、RolloutPoison、CounterTrap、ArgmaxQTrap。
- Web UI：可以人工对战 AI，也可以观察不同 agent 行为。
- Arena：可以并行跑比赛、断点续跑、输出 CSV、matchup matrix 和 score matrix。
- AlphaZero 工程接口：已有 policy/value 网络、PUCT 接口、teacher bootstrap、batched self-play 入口。

重要判断：课程项目的核心代码已经够了。现在最缺的是正式实验数据、结果表和论文。

## 3. 课程交付最重要的下一步

优先级最高的是跑正式实验，不是继续无止境增强 AlphaZero。

### Step A: 跑 targeted trap 实验

这个实验直接回答 proposal 的核心问题：每个 trap 能不能打败它对应的 victim。

```powershell
python experiments\run_trap_targets.py --games-per-pair 100 --max-turns 120 --workers 4 --resume --output experiments\results\final_trap_targets_games.csv --matrix-output experiments\results\final_trap_targets_matchups.csv --score-matrix-output experiments\results\final_trap_targets_scores.csv --summary-output experiments\results\final_trap_effectiveness.csv
```

输出文件：

- `experiments\results\final_trap_targets_games.csv`
- `experiments\results\final_trap_targets_matchups.csv`
- `experiments\results\final_trap_targets_scores.csv`
- `experiments\results\final_trap_effectiveness.csv`

论文里优先引用 `final_trap_effectiveness.csv` 和 matchup matrix。

### Step B: 跑 research tournament

这个实验看所有主要 agent 之间的整体胜率关系。

```powershell
python experiments\run_tournament.py --preset research --games-per-pair 20 --max-turns 150 --workers 4 --resume --output experiments\results\final_research_games.csv --matrix-output experiments\results\final_research_matchups.csv --score-matrix-output experiments\results\final_research_scores.csv
```

输出文件：

- `experiments\results\final_research_games.csv`
- `experiments\results\final_research_matchups.csv`
- `experiments\results\final_research_scores.csv`

如果机器很慢，先把 `--games-per-pair 20` 改成 `5` 做 smoke test，确认能跑通后再恢复正式参数。

### Step C: 跑 ablation

这个实验解释 trap 参数是否真的有影响。

```powershell
python experiments\run_ablation.py --weights 0,2,4,8,12 --games 50 --max-turns 120 --output experiments\results\final_path_lure_ablation.csv
```

输出文件：

- `experiments\results\final_path_lure_ablation.csv`

论文里可以用它说明 PathLure 不是随机运气，而是参数变化会影响效果。

## 4. RL/NN 部分怎么处理

RL/NN 是 proposal 的扩展部分，不应该阻塞主线。

建议先训练比较稳定的 Q-learning 和 Approx-Q：

```powershell
python experiments\train_q_learning.py --episodes 2000 --max-turns 120 --output experiments\results\q_learning_policy_final.json
python experiments\train_approx_q.py --episodes 2000 --max-turns 120 --output experiments\results\approx_q_policy_final.json
```

然后再跑 targeted trap 或 research tournament，观察 ArgmaxQTrap 是否明显变强。

Deep-Q 和 AlphaZero 可以写成 extension。不要在论文里声称 AlphaZero 已经达到强棋力，除非后面真的有 arena gating 结果支持。

## 5. AlphaZero 当前状态

AlphaZero 相关代码现在属于工程基础设施：

- `experiments/train_alphazero_teacher_bootstrap.py`：用强启发式 teacher 生成初始训练数据。
- `experiments/run_batched_selfplay.py`：用 batched MCTS / batched inference 生成 self-play 数据。
- `experiments/run_alphazero_config.py`：按 JSON 配置跑长训练。

如果要继续远端训练，先跑小规模 batched self-play smoke：

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python experiments/run_batched_selfplay.py --games 128 --workers 4 --simulations 16 --max-turns 120 --hidden-size 256 --action-limit 14 --wall-limit 7 --mcts-batch-size 8 --batch-size 256 --epochs 2 --device cuda --initial-checkpoint experiments/results/alphazero_teacher_bootstrap.pt --examples-output experiments/results/batched_selfplay_128_examples.pt --train-output experiments/results/alphazero_batched_selfplay_128.pt
```

这个方向是加分项，不是课程项目能否完成的关键路径。

## 6. 论文应该怎么写

论文建议结构：

1. Introduction：说明我们不是追求最强 agent，而是研究如何 exploit baseline blind spots。
2. Game and Baselines：介绍 Quoridor、Greedy BFS、Minimax、MCTS、RL/NN baselines。
3. Adversarial Policies：分别解释 PathLure、DepthTrap、RolloutPoison、CounterTrap、ArgmaxQTrap。
4. Experiments：说明 arena 设置、games-per-pair、max-turns、seeds、metrics。
5. Results：放 targeted trap table、research matchup matrix、ablation table。
6. Discussion：解释哪些 trap 成功，哪些不稳定，为什么。
7. Limitations：说明 Deep-Q/AlphaZero 还没有充分训练，结果不能夸大。

不要手写胜率。所有表格都要从 `experiments\results\*.csv` 来。

## 7. 同学分工建议

最容易分工的方式：

- 同学 A：跑 `final_trap_targets`，检查 CSV 是否完整。
- 同学 B：跑 `final_research`，整理 matchup matrix。
- 同学 C：跑 ablation，并画一张参数变化图。
- 同学 D：写 Methods，解释每个 trap 的机制。
- 同学 E：写 Results / Discussion，只引用 CSV 中已有结果。

每个人提交结果时都要写清楚：

- 运行的命令。
- 输出的 CSV 文件路径。
- 是否使用 `--resume`。
- 是否有 error row 或 max-turn draw 过多的问题。

## 8. 常见问题

如果实验中断：

```powershell
重新运行同一条命令，保留 --resume
```

如果速度太慢：

```powershell
先把 --games-per-pair 降低到 5 做测试
```

如果结果里 draw 太多：

```powershell
提高 --max-turns，或者在论文里把 draw rate 当作一个现象报告
```

如果某个 trap 没有效果：

不要隐藏它。论文可以写成“该 victim 的弱点没有被当前 minimal policy 稳定利用”，这也是有效结果。

## 9. 当前最重要结论

当前代码已经可以支持课程项目完成。后续最重要的不是继续写更多 agent，而是：

1. 跑正式实验。
2. 固定结果 CSV。
3. 从 CSV 生成论文表格。
4. 把成功和失败都诚实写进论文。
