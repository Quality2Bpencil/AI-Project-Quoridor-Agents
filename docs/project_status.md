# 项目目标与进度记录

更新时间：2026-06-26

## 项目定位

CS181 final project：

**From Developing to Exploiting Quoridor Agents: Minimal Adversarial Policies Against Search and Learning Baselines**

项目重点不是堆叠“更强的通用 Quoridor agent”，而是研究一个更窄、更可解释的问题：

> 如何用较简单的 adversarial policy，稳定利用 Greedy BFS、depth-limited Minimax、finite-budget MCTS，以及后续 deterministic RL / NN victim 的盲点？

Quoridor 的墙放置可以看成对对手最短路图的 adversarial cut；同时每一步的合法墙检查必须保证双方仍有通路。因此这个游戏天然适合研究 search agent 在局部评估、有限深度、有限 rollout 和确定性 policy 下的可利用弱点。

## Proposal 核心目标

proposal 中的研究路线可以拆成四层：

1. 建立标准 Quoridor 环境：9x9 棋盘、双方各 10 堵墙、移动/跳跃/侧跳、墙冲突、不能完全封死路径。
2. 实现传统 baseline：Greedy BFS、Minimax + Alpha-Beta、standard/finite-budget MCTS。
3. 设计 minimal adversarial policies：
   - Path-Lure：利用 Greedy BFS 只看最短路长度、忽略 path diversity。
   - Depth-Trap：利用 depth-limited Minimax 的 horizon 问题。
   - Rollout-Poison：利用有限预算 MCTS 在浅 rollout 下无法区分 deceptive response。
4. 扩展到 deterministic learning baseline：
   - Argmax RL victim。
   - Neural-pruned MCTS / PUCT。
   - Approximate Q-Learning / Deep Q victim。

当前仓库已经完成第 1-3 层，并补齐了第 4 层的可运行初版：tabular Q-learning、linear Approx-Q、Torch Deep-Q、Argmax-Q victim trap，以及 heuristic-prior PUCT。真正神经网络 policy-value PUCT 仍未实现。

## 当前已实现

| 模块 | 状态 | 位置 | 说明 |
| --- | --- | --- | --- |
| Quoridor 核心规则 | 已完成 | `quoridor/core/` | 包含状态、动作、合法移动、合法放墙、通路检查、胜负判断。 |
| 环境封装 | 已完成 | `quoridor/core/env.py` | `QuoridorEnv.step()` 返回状态、reward、done、winner。 |
| Greedy BFS baseline | 已完成 | `quoridor/agents/greedy_bfs.py` | 基于最短路启发式的一步贪心。 |
| Minimax baseline | 已完成 | `quoridor/agents/minimax.py` | depth-limited alpha-beta minimax，使用候选动作剪枝。 |
| MCTS baseline | 已完成 | `quoridor/agents/mcts.py` | finite-budget UCT + shallow heuristic rollout。 |
| PathLureAgent | 已完成 | `quoridor/agents/adversarial.py` | 针对 Greedy BFS 的 path diversity trap。 |
| DepthTrapAgent | 已完成 | `quoridor/agents/adversarial.py` | 针对 shallow/depth-limited minimax 的 follow-up trap。 |
| RolloutPoisonAgent | 已完成 | `quoridor/agents/adversarial.py` | 针对 finite-budget MCTS 的 response ambiguity。 |
| CounterfactualTrapAgent | 已完成初版 | `quoridor/agents/adversarial.py` | 对多个 plausible victim response 做鲁棒打分，奖励 path delta + trap transition + follow-up。 |
| ArgmaxQTrapAgent | 已完成初版 | `quoridor/agents/adversarial.py` | 用 deterministic `QLearningAgent(epsilon=0)` 作为 victim response model，补齐 argmax RL exploitation 接口。 |
| 共享启发式 | 已完成 | `quoridor/agents/heuristics.py` | path distance、path diversity、candidate action ranking。 |
| PUCT baseline | 已完成初版 | `quoridor/agents/puct.py` | heuristic policy prior + value estimate 的 PUCT，可替换为训练后的 neural prior/value。 |
| Tournament / Elo | 已完成轻量版 | `quoridor/evaluation/`, `experiments/run_tournament.py` | 支持 round-robin、CSV、Elo standings。 |
| Formal Arena | 已完成初版 | `quoridor/evaluation/arena.py`, `experiments/run_tournament.py` | 支持并行 worker、resume、增量 CSV、matchup matrix、score matrix。 |
| PathLure ablation | 已完成轻量版 | `experiments/run_ablation.py` | 支持 trap weight sweep。 |
| 训练接口 | 已完成接口 | `quoridor/training/discrete_env.py` | 209 维离散动作空间、legal mask、flat observation。 |
| Tabular Q-Learning | 已完成初版 | `quoridor/agents/q_learning.py`, `quoridor/training/q_learning.py` | 自博弈 Q-table、JSON policy 保存/加载、UI/tournament agent 接入。 |
| Linear Approx-Q | 已完成初版 | `quoridor/agents/approx_q.py`, `quoridor/training/approx_q_learning.py` | 线性特征 Q-function、自博弈权重训练、JSON weight 保存/加载。 |
| Deep Q-Network | 已完成初版 | `quoridor/agents/deep_q.py`, `quoridor/training/deep_q.py` | PyTorch DQN、target net、replay buffer、legal action mask、CUDA 训练入口。 |
| Tkinter UI | 已完成基础版 | `visual.py`, `quoridor/ui/` | 支持 Human/Random 组合。 |
| Web UI | 已完成展示版 | `visual_web.py`, `quoridor/web/` | 支持 Human/Agent 任意组合、3D 棋盘、像素风 HUD、合法动作提示、agent step/play。 |
| 单元测试 | 已完成基础覆盖 | `tests/` | 覆盖规则、训练接口、search agents、evaluation、web session。 |

## Web UI 视觉与交互记录

当前 Web UI 的美术方向是参考 `tmp/new_ref.png` 的复古游戏化 3D 棋盘，采用更接近成熟游戏 NPR 管线的拆分：

- 棋子光影：使用 `MeshToonMaterial` 和 nearest-filter toon ramp，并对 OBJ 棋子做顶点合并后重算法线，减少三角面和模型接缝造成的碎裂光影。
- 棋子外轮廓：不再依赖 Three.js `OutlinePass` 或 inverted hull / BackSide shell。当前使用专用 screen-space silhouette mask：先把棋子单独渲染到 mask，再膨胀固定屏幕像素并减去原 mask，只保留外侧黑色轮廓。
- 全屏后处理：关闭全屏亮度边缘线，避免把棋子内部纹理或 toon 色阶误画成黑色碎边；保留轻量色阶化和暗角。
- 墙体交互：墙模式下显示一个跟随鼠标移动的真实 3D 墙体 preview，自动吸附到棋盘墙槽；合法位置显示绿色底座，非法位置显示 muted warning，并通过正常 `/api/human-action` 流程落子。
- 木质结构：棋盘和墙体使用程序化木纹贴图，墙体额外使用显式黑色边线，避免依赖模糊/色块来制造复古感。

## 当前未实现 / 仍需补齐

| 目标 | 状态 | 需要补的内容 |
| --- | --- | --- |
| Argmax RL victim exploitation | 已完成初版，待正式实验 | 已有 deterministic Q victim 和 `ArgmaxQTrapAgent`；仍需训练稳定 Q-table 并跑 exploit 对照。 |
| Approximate Q-Learning / Deep Q victim | 已完成初版，待正式实验 | 已有 linear feature Q-function、Torch DQN、训练脚本和 checkpoint 加载 agent。 |
| Neural-pruned MCTS / PUCT | heuristic-prior PUCT 已完成，neural prior/value 未实现 | 已有 PUCT search 接口；后续训练 policy/value model 后可替换默认启发式 prior/value。 |
| 1000+ rounds 正式实验 | 未完成 | 当前只有轻量 tournament/ablation 脚本；需要固定 seeds、参数表、正式 CSV 和汇总。 |
| 论文/报告级结果表 | 部分完成 | Arena 已可输出 matchup matrix 和 score matrix；仍需正式长跑数据。 |
| 学术论文稿件 | 已完成骨架 | `docs/paper_draft.md` 已建立论文结构、方法映射、实验待办和未验证结果占位。 |
| Web UI 离线资源 | 部分完成 | pawn 模型已本地化；Three.js 仍从 CDN 加载，完全离线展示需要 vendored JS 或构建流程。 |

## 当前验证状态

最近一次验证命令：

```powershell
node --check quoridor\web\static\app.js
python -m unittest discover -s tests -v
```

验证结果：

- JavaScript 语法检查通过。
- Python 单元测试通过，当前测试数为 58。
- Web UI 已在本地 `http://127.0.0.1:8766` 做过桌面和移动视口 smoke check。

## 算法审计记录

2026-06-25 先审计现有算法，再进入新算法设计。当前结论：

- 核心规则、训练接口、Web session、Greedy BFS、Minimax、MCTS、PathLure、DepthTrap、RolloutPoison 的单元测试全部通过。
- `experiments/run_tournament.py --preset smoke --games-per-pair 1 --max-turns 30` 已通过并写出 CSV。
- `experiments/run_tournament.py --preset adversarial --games-per-pair 1 --max-turns 5` 已通过并写出 CSV，但全量 adversarial preset 比较慢，后续正式实验前需要单独做性能预算和参数表。
- 修正了 evaluation 中的 `trap_events` 统计语义：现在只统计对手从非陷阱状态进入陷阱状态的转移，不再在陷阱持续存在时每回合重复计数。

本轮新增验证命令：

```powershell
python -m compileall -q quoridor tests experiments
python -m unittest discover -s tests -v
python experiments\run_tournament.py --preset smoke --games-per-pair 1 --max-turns 30 --output tmp\algorithm_audit_tournament_smoke_after.csv
python experiments\run_tournament.py --preset adversarial --games-per-pair 1 --max-turns 5 --output tmp\algorithm_audit_tournament_adversarial_tiny.csv
```

本轮 RL / PUCT 验证命令：

```powershell
python -m unittest tests.test_approx_q tests.test_puct tests.test_q_learning tests.test_search_agents -v
python experiments\train_q_learning.py --episodes 3 --max-turns 8 --output tmp\q_learning_smoke_policy.json
python experiments\train_approx_q.py --episodes 3 --max-turns 8 --output tmp\approx_q_smoke_policy.json
python experiments\run_tournament.py --preset full --games-per-pair 1 --max-turns 20 --output tmp\tournament_full_after_algorithms.csv
python experiments\run_tournament.py --preset adversarial --games-per-pair 1 --max-turns 5 --output tmp\tournament_adversarial_after_algorithms.csv
```

结果：

- `python -m compileall -q quoridor tests experiments` 通过。
- 新增算法相关 23 个测试通过。
- 全量单元测试 58 个通过。
- Q-learning smoke training：3 episodes，写出 `tmp\q_learning_smoke_policy.json`。
- Approx-Q smoke training：3 episodes，写出 `tmp\approx_q_smoke_policy.json`。
- `full` tournament smoke 通过，写出 `tmp\tournament_full_after_algorithms.csv`。
- `adversarial` tournament smoke 通过，写出 `tmp\tournament_adversarial_after_algorithms.csv`。
- `research` preset 默认参数较重，`--max-turns 5` 也不适合日常 smoke；正式实验前需要先降低或拆分参数预算。
- Web server 已重启在 `http://127.0.0.1:8766`，当前新增 agent 单步 API 粗测：
  - `PUCT 8` avg 123.5ms, max 333.4ms。
  - `Approx-Q` avg 14.9ms, max 28.8ms。
  - `ArgmaxQTrap` avg 85.2ms, max 218.7ms。

## GPU 训练记录

本机 GPU：

- NVIDIA GeForce RTX 4060 Laptop GPU, 8GB VRAM。

结果：

- Tabular Q-learning：200 episodes，244.06s，wins `(125, 74)`, draws `1`，写出 `experiments\results\q_learning_policy.json`。
- Approx-Q：200 episodes，wins `(93, 99)`, draws `8`，写出 `experiments\results\approx_q_policy.json`。
- Deep-Q：100 episodes，230.72s，device `cuda`，updates `11107`，wins `(8, 13)`, draws `79`，写出 `experiments\results\deep_q_policy.pt`。

## Formal Arena 记录

现有正式竞技场入口：

```powershell
python experiments\run_tournament.py --preset smoke --games-per-pair 2 --max-turns 6 --workers 2 --resume --output tmp\arena_smoke_games.csv --matrix-output tmp\arena_smoke_matrix.csv --score-matrix-output tmp\arena_smoke_score.csv
```

能力：

- 并行：`--workers N` 使用 `ProcessPoolExecutor` 多进程执行每局 game task。
- 可恢复：`--resume` 会读取已有 output CSV 的 `game_id`，跳过已完成局。
- 增量落盘：每局完成后由主进程 append 一行 CSV，并 flush，避免长跑中断丢全部结果。
- 去重随机性：新 arena row 包含 `seed`，preset worker 会用每局 seed offset 构建 agent，避免把同一 deterministic 对局重复统计成多局。
- 性能记录：新 arena row 包含 `elapsed_seconds`，matchup matrix 包含 `avg_elapsed_seconds`。
- 进度输出：`--progress-interval N` 每 N 局输出完成数、吞吐率和 ETA。
- matchup matrix：`--matrix-output` 输出 long-format 对阵矩阵，包含 games/wins/losses/draws/score_rate/win_rate/avg_turns/avg_trap_events/avg_wall_actions/avg_path_delta。
- score matrix：`--score-matrix-output` 输出 wide-format 分数率矩阵，适合直接转论文表格或 heatmap。

验证：

- 第一次并行 smoke：`tasks=6 completed_now=6 skipped_or_existing=0 workers=2`。
- 第二次同命令 resume：`tasks=6 completed_now=0 skipped_or_existing=6 workers=2`。
- 已生成 `tmp\arena_smoke_games.csv`、`tmp\arena_smoke_matrix.csv`、`tmp\arena_smoke_score.csv`。
- 全量单元测试：62 tests OK。

注意：

- `research` preset 含 MCTS/PUCT/CounterTrap/ArgmaxQTrap/Deep-Q，建议先用 `--workers 2` 或 `--workers 4` 试跑；Deep-Q 多进程会让每个 worker 各自加载 checkpoint，workers 过多会浪费显存。
- 正式论文实验建议把 games CSV、matchup matrix 和 score matrix 一起保存到 `experiments\results\`，不要只保存 standings 文本输出。

## Trap Efficacy 竞技记录

为论文准备的第一批 trap efficacy 数据已经生成。早期 deterministic 数据保留作开发记录；正式引用优先使用带 per-game seed 和置信区间的 optimized target 数据。

完整 targeted preset round-robin：

```powershell
python experiments\run_tournament.py --preset trap_eval --games-per-pair 6 --max-turns 80 --workers 4 --resume --output experiments\results\trap_eval_games.csv --matrix-output experiments\results\trap_eval_matchups.csv --score-matrix-output experiments\results\trap_eval_scores.csv
```

结果：

- 216 / 216 局完成，全部 `status=ok`。
- 输出：
  - `experiments\results\trap_eval_games.csv`
  - `experiments\results\trap_eval_matchups.csv`
  - `experiments\results\trap_eval_scores.csv`
  - `experiments\results\trap_effectiveness.csv`

为了更直接衡量 intended trap-victim pair，又跑了专门目标对局：

```powershell
python experiments\run_trap_targets.py --games-per-pair 20 --max-turns 100 --workers 4 --resume --output experiments\results\trap_targets_games.csv --matrix-output experiments\results\trap_targets_matchups.csv --score-matrix-output experiments\results\trap_targets_scores.csv --summary-output experiments\results\trap_effectiveness_targets.csv
```

结果表：

| Trap | Target | Games | Wins | Losses | Score rate | Avg trap events | Avg target path delta | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PathLure | Greedy BFS | 20 | 10 | 10 | 0.5000 | 0.000 | -5.000 | 当前版本未证明有效。 |
| DepthTrap | Minimax d1 | 20 | 20 | 0 | 1.0000 | 0.000 | -2.500 | 有效。 |
| RolloutPoison | MCTS 5 | 20 | 20 | 0 | 1.0000 | 1.000 | -0.500 | 有效。 |
| CounterTrap | MCTS 5 | 20 | 20 | 0 | 1.0000 | 0.500 | 3.500 | 有效。 |
| ArgmaxQTrap | Q-learning | 20 | 10 | 10 | 0.5000 | 1.000 | 2.500 | 当前版本未证明有效，需要改进或更多训练。 |

论文表格源文件：

- `experiments\results\trap_targets_games.csv`
- `experiments\results\trap_targets_matchups.csv`
- `experiments\results\trap_targets_scores.csv`
- `experiments\results\trap_effectiveness_targets.csv`

初步结论必须保守写：目前能证明 **DepthTrap、RolloutPoison、CounterTrap** 对目标 victim 有效；**PathLure、ArgmaxQTrap** 当前实现还不能作为有效 trap 结论。

优化后 seeded target 对局：

```powershell
python experiments\run_trap_targets.py --games-per-pair 20 --max-turns 100 --workers 4 --progress-interval 20 --output experiments\results\trap_targets_optimized_games.csv --matrix-output experiments\results\trap_targets_optimized_matchups.csv --score-matrix-output experiments\results\trap_targets_optimized_scores.csv --summary-output experiments\results\trap_effectiveness_optimized_targets.csv
```

结果表：

| Trap | Target | Games | Wins | Losses | Score rate | 95% CI | Avg trap events | Avg target path delta | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| PathLure | Greedy BFS | 20 | 20 | 0 | 1.0000 | [0.8389, 1.0000] | 0.000 | -7.000 | 胜率有效，但当前 trap event 指标未捕捉到路径陷阱机制。 |
| DepthTrap | Minimax d1 | 20 | 19 | 1 | 0.9500 | [0.7639, 0.9911] | 0.300 | 0.350 | 有效。 |
| RolloutPoison | MCTS 5 | 20 | 18 | 2 | 0.9000 | [0.6990, 0.9721] | 0.400 | 0.750 | 有效。 |
| CounterTrap | MCTS 5 | 20 | 19 | 1 | 0.9500 | [0.7639, 0.9911] | 0.300 | 2.250 | 有效。 |
| ArgmaxQTrap | Q-learning | 20 | 13 | 6 | 0.6750 | [0.4567, 0.8369] | 1.150 | 0.200 | 有一定信号，但样本量下仍需更多训练/对局确认。 |

优化内容：

- Arena 每局使用 `seed` offset，不再把固定 seed 的同一盘重复统计为多局。
- Arena 记录 `elapsed_seconds`，matrix 输出 `avg_elapsed_seconds`。
- PathLure 增加 bad-wall safety gate，避免墙没有增加目标路径或降低 path diversity 时仍过度放墙。
- ArgmaxQTrap 默认改为 deterministic victim response (`response_width=1`) 并提高 path-delta/follow-up 权重。

正式论文优先引用：

- `experiments\results\trap_targets_optimized_games.csv`
- `experiments\results\trap_targets_optimized_matchups.csv`
- `experiments\results\trap_targets_optimized_scores.csv`
- `experiments\results\trap_effectiveness_optimized_targets.csv`

## 新算法设计进展

在现有算法稳定后，已开始实现一个仍然保持 minimal adversarial policy 主线的算法：**Counterfactual Response Trap Search**。

核心想法：

1. 对每个候选动作先模拟对手最可能的 1-2 个 response。
2. 只奖励能让对手路径多样性从高变低、且最短路变长的“陷阱转移”，避免把已经存在的困境重复计分。
3. 在对手 response 后再看我方一个 follow-up 的收益，用低深度搜索近似 horizon trap。
4. 对不同 victim 使用同一框架，只替换 response model：
   - Greedy BFS victim：用当前 `GreedyBFSAgent`。
   - Minimax victim：用浅层 `MinimaxAgent` 或 `ranked_actions` 近似。
   - MCTS victim：用小预算 MCTS 或 rollout score gap 近似。

这个方向比直接上 NN/RL 更贴近 proposal 的“minimal adversarial policies”主题，也能复用当前已经验证过的 path diversity、path delta、trap transition 指标。

当前实现名为 `CounterfactualTrapAgent`，并已接入：

- Python agent 导出：`quoridor.agents.CounterfactualTrapAgent`
- Web UI agent 选项：`CounterTrap`
- CLI adversarial / research preset：`counter_trap`
- 单元测试：`test_counterfactual_trap_returns_legal_action`

初步验证：

```powershell
python -m unittest tests.test_search_agents tests.test_web_server -v
python experiments\run_tournament.py --preset adversarial --games-per-pair 1 --max-turns 5 --output tmp\counter_trap_adversarial_tiny.csv
```

注意：加入 `counter_trap` 后 adversarial preset 的组合数和每步计算都会增加，正式实验前需要固定参数预算。

### RL / PUCT 补齐进展

本轮新增内容：

- `QLearningAgent`：tabular Q-learning victim，使用 current-player perspective state key，未训练或未覆盖状态会回退到一层启发式。
- `ApproxQLearningAgent`：linear function approximation，特征包括 path progress、opponent slowdown、wall/move 类型、wall balance、center file 和 terminal flag。
- `DeepQAgent`：PyTorch DQN checkpoint agent，训练时使用 replay buffer、target network 和 legal action mask，可在 CUDA 环境下运行。
- `ArgmaxQTrapAgent`：把 deterministic Q policy 作为 victim response model，复用 `CounterfactualTrapAgent` 的 robust response scoring。
- `PUCTAgent`：使用 heuristic policy prior 和 normalized heuristic value 的 PUCT，当前是 neural-pruned MCTS 的接口级 baseline，不声称已有训练后的 neural network。
- 训练脚本：
  - `experiments/train_q_learning.py`
  - `experiments/train_approx_q.py`

当前策略与 proposal 的对应关系：

| Proposal 方向 | 当前实现 | 说明 |
| --- | --- | --- |
| Trap against Greedy BFS | `PathLureAgent` | 利用最短路贪心忽略 path diversity。 |
| Trap against depth-limited Minimax | `DepthTrapAgent` | 利用 shallow search 的 follow-up horizon。 |
| Trap against finite-budget MCTS | `RolloutPoisonAgent`, `CounterfactualTrapAgent` | 利用 shallow rollout / response ambiguity。 |
| Trap against argmax RL victim | `ArgmaxQTrapAgent` | 使用 deterministic `QLearningAgent` 作为 victim。 |
| Approximate Q-Learning | `ApproxQLearningAgent` | 线性近似版本。 |
| Deep Q victim | `DeepQAgent` | Torch DQN，已支持 CUDA 训练和 checkpoint 加载。 |
| Neural-pruned MCTS / PUCT | `PUCTAgent` | 已实现 PUCT search；neural prior/value 尚未训练。 |

## 推荐下一步

优先级建议如下：

1. 跑正式 adversarial preset：

   ```powershell
   python experiments\run_tournament.py --preset adversarial --games-per-pair 10 --max-turns 150 --output experiments\results\tournament_adversarial.csv
   ```

2. 跑 PathLure ablation：

   ```powershell
   python experiments\run_ablation.py --weights 0,2,4,8,12 --games 20 --max-turns 150 --output experiments\results\path_lure_ablation.csv
   ```

3. 为正式报告整理 summary table：
   - win rate
   - Elo
   - avg trap events
   - avg opponent min path diversity
   - avg opponent path delta
   - disqualification / max-turn draw rate

4. 如果要继续 proposal 的 NN/RL 部分，先实现 deterministic Q victim，再写 exploit wrapper。不要直接跳到复杂 NN-MCTS，否则很难和当前 minimal adversarial policy 主线对齐。

5. 下一步正式实验前，先训练并冻结两个 policy 文件：

   ```powershell
   python experiments\train_q_learning.py --episodes 500 --max-turns 120 --output experiments\results\q_learning_policy.json
   python experiments\train_approx_q.py --episodes 500 --max-turns 120 --output experiments\results\approx_q_policy.json
   F:\Programs\PythonEnv\torch10\python.exe experiments\train_deep_q.py --episodes 500 --max-turns 120 --device cuda --output experiments\results\deep_q_policy.pt
   ```

6. 论文结果表应从固定输出生成，不手写结果：

   ```powershell
   python experiments\run_tournament.py --preset research --games-per-pair 10 --max-turns 150 --workers 4 --resume --output experiments\results\tournament_research_games.csv --matrix-output experiments\results\tournament_research_matchups.csv --score-matrix-output experiments\results\tournament_research_scores.csv
   ```

## 交接备注

- 新 agent 只需要实现 `choose_action(state, legal_actions)`。
- 规则引擎已经比较完整，除非发现规则 bug，不建议在实验阶段改 `quoridor/core/`。
- 搜索类 agent 的性能关键在候选墙位筛选，不要在每一步完整枚举所有墙再深搜。
- Web UI 是展示和调试工具，不是训练入口；训练接口以 `DiscreteQuoridorEnv` 为准。

## 启发式增强与 AlphaZero 路线

本轮针对“人类直走即可获胜”和 DepthTrap 互相横跳的问题，开始把算法分成两层：

1. **强启发式搜索层**：作为当前 Web 可玩 AI、arena baseline 和未来 AlphaZero teacher。
2. **AlphaZero-style policy/value 层**：作为最终强 AI 的训练方向。

### 已完成的启发式增强

`quoridor/agents/heuristics.py` 的局面评分从单一 shortest-path 差值扩展为可解释分量：

| Term | 目的 |
| --- | --- |
| `path_distance` | 最短路差值，衡量 race 基础优势。 |
| `wall_balance` | 墙资源差，避免无意义过度放墙。 |
| `path_diversity` | 最短路第一步选择数差，衡量是否被 funnel/trap。 |
| `pawn_mobility` | 当前棋子可移动性差，衡量局部封锁。 |
| `goal_progress` | 棋子实际推进度，补充最短路相同但位置不同的情况。 |
| `tempo` | 当前行动权的小权重。 |
| `pawn_race` | 开局 race projection，用于识别双方直线冲刺时的跳子反杀。 |

墙动作评分新增：

- 对手最短路增量奖励。
- 对手 path diversity 下降奖励。
- 对手 pawn mobility 下降奖励。
- 靠近对手前进方向和中心线的 positional bonus。
- 空 tempo wall 的轻惩罚。

`MCTSAgent` 与 `PUCTAgent` 增加 root tactical shortcut：当启发式明确判断需要放墙阻止 losing pawn race 时，直接返回该墙，不再在 Web 端浪费模拟预算。

学习策略新增 safety fallback：

- `QLearningAgent`
- `ApproxQLearningAgent`
- `DeepQAgent`

当模型动作显著差于启发式动作时，Web/arena 默认会使用启发式动作兜底，避免早期弱 checkpoint 出现明显低级失误。

### AlphaZero 工程接口

新增 AlphaZero-style 组件：

- `quoridor/agents/alphazero.py`
  - `AlphaZeroAgent`
  - 有 checkpoint 时使用 neural policy/value + PUCT。
  - 无 checkpoint 时默认不可用；Web 下拉会禁用该选项，而不是自动 fallback。
- `quoridor/training/alphazero.py`
  - `AlphaZeroNet`
  - `AlphaZeroExample`
  - `policy_vector`
  - `alphazero_loss`
  - `save_alphazero_checkpoint`
  - `load_alphazero_checkpoint`
- `PUCTAgent.search_policy(...)`
  - 输出 visit-count policy distribution，用于 AlphaZero 自对弈训练 target。
- `experiments/train_alphazero.py`
  - 运行小规模 AlphaZero-style self-play 训练并保存 checkpoint。

### 参考成熟做法

本项目的最终路线参考：

- AlphaGo Zero：self-play 生成训练目标，单一网络预测 policy 与 value，再用 MCTS/PUCT 改善行动选择。
- AlphaZero：去除手工 domain-specific augmentations，使用通用 policy/value + MCTS 框架迁移到 chess/shogi/Go。
- MCTS survey / heuristic MCTS：纯随机 rollout 在复杂棋类里通常不够，需要 heuristic rollout、prior、implicit minimax 或 value 初始化。
- Quoridor MCTS 相关研究：Quoridor 状态空间和 game-tree complexity 很高，有限预算 MCTS 必须结合领域启发式和候选动作剪枝。

### 终极 AI 训练路线

建议分三阶段推进：

1. **Teacher-guided bootstrap**
   - 用当前 enhanced heuristic PUCT / MCTS 生成初始 `(state, pi, z)`。
   - 训练 `AlphaZeroNet` 先模仿 stronger search，避免从随机网络开始时样本效率太低。

2. **Self-play AlphaZero**
   - 每步使用 `PUCTAgent.search_policy` 得到 visit-count policy。
   - 前若干回合使用 temperature > 0 增加探索，后期接近 argmax。
   - 终局后把胜负 `z` 回填到整局样本。
   - 用 `alphazero_loss = value MSE + policy cross entropy` 更新网络。

3. **Arena gating**
   - 新 checkpoint 必须在正式 arena 中击败当前 best checkpoint 和 enhanced heuristic baseline。
   - 通过后才更新 `experiments/results/alphazero_policy_value.pt`。

短期不要声称已经实现“真正 AlphaZero 强度”。当前完成的是 AlphaZero-compatible inference/training interface；正式强度还取决于大规模自对弈训练和 arena gating。

### AlphaZero 训练启动记录

已完成一轮本地 CUDA smoke training：

```powershell
F:\Programs\PythonEnv\torch10\python.exe experiments\train_alphazero.py --games 2 --simulations 4 --max-turns 30 --hidden-size 64 --action-limit 6 --wall-limit 3 --batch-size 8 --epochs-per-game 1 --seed 0 --device cuda --output experiments\results\alphazero_policy_value.pt
```

结果：

- games: 2
- examples: 60
- updates: 2
- wins: `(0, 0)`
- draws: 2
- device: `cuda`
- elapsed_seconds: 18.554

该 checkpoint 只证明 self-play/training/checkpoint/Web 启用链路可运行，不代表已有强棋力。

### 远程 AlphaZero 长训练配置

已新增正式远程长训练配置：

- 配置：`experiments/configs/alphazero_remote_long.json`
- Runner：`experiments/run_alphazero_config.py`
- 默认最终输出：`experiments/results/alphazero_policy_value.pt`
- 分阶段 checkpoint 目录：`experiments/results/alphazero_stages/`

当前配置不使用本地 2 局 smoke checkpoint 作为起点，而是从零开始训练一个干净的 256 hidden-size policy/value 网络。默认阶段：

| Stage | Games | Chunk | Simulations | Max turns | Action / wall budget | Batch | Replay cap | 目的 |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| remote_validation | 64 | 16 | 8 | 120 | 10 / 5 | 64 | 20,000 | 验证远程 CUDA、依赖、checkpoint 恢复和吞吐。 |
| bootstrap | 512 | 64 | 16 | 140 | 14 / 7 | 128 | 60,000 | 初步摆脱随机 policy，学习基本 race / block 模式。 |
| policy_improvement | 4,096 | 128 | 48 | 170 | 22 / 10 | 256 | 160,000 | 形成可对抗 heuristic PUCT/MCTS 的候选策略。 |
| champion_search | 16,384 | 256 | 96 | 190 | 30 / 14 | 512 | 320,000 | 长跑候选冠军 checkpoint。 |

远程 Ubuntu 建议启动命令：

```bash
cd /path/to/Quoridor
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy
export PYTHONPATH="$PWD"
python experiments/run_alphazero_config.py --config experiments/configs/alphazero_remote_long.json --resume
```

先做 dry run：

```bash
python experiments/run_alphazero_config.py --config experiments/configs/alphazero_remote_long.json --dry-run
```

如果远程 GPU/CPU 比预期慢，先只跑验证阶段：

```bash
python experiments/run_alphazero_config.py --config experiments/configs/alphazero_remote_long.json --resume --max-stages 1
```

训练晋级标准不写成“主观强”，而写成可复现实验门槛：

1. 候选 checkpoint 必须先通过 `remote_validation` 并能被 Web 的 `AlphaZero` agent 加载。
2. 候选 checkpoint 进入 arena gating，至少对 `PUCT 64`、`MCTS 64`、`DepthTrap`、`CounterTrap` 和上一版 best AlphaZero 达到配置中的 score-rate 阈值。
3. 通过 gating 后才允许覆盖 `experiments/results/alphazero_policy_value.pt` 作为 Web 端可选强策略。

注意：当前 runner 的可恢复粒度是 chunk checkpoint；中断后用 `--resume` 跳过已完成 chunk。为控制内存，self-play replay buffer 使用 `replay_capacity` 裁剪，统计中的 `examples` 是累计生成样本数，不是当前内存中保留的样本数。
