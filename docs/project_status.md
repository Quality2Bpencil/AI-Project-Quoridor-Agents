# 项目目标与进度记录

更新时间：2026-06-25

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

当前仓库主要完成了第 1-3 层；第 4 层还没有实际训练代码或可复现实验。

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
| 共享启发式 | 已完成 | `quoridor/agents/heuristics.py` | path distance、path diversity、candidate action ranking。 |
| Tournament / Elo | 已完成轻量版 | `quoridor/evaluation/`, `experiments/run_tournament.py` | 支持 round-robin、CSV、Elo standings。 |
| PathLure ablation | 已完成轻量版 | `experiments/run_ablation.py` | 支持 trap weight sweep。 |
| 训练接口 | 已完成接口 | `quoridor/training/discrete_env.py` | 209 维离散动作空间、legal mask、flat observation。 |
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
| Argmax RL victim exploitation | 未实现 | 需要训练或加载一个 deterministic Q policy，并暴露 `choose_action(state, legal_actions)`。 |
| Approximate Q-Learning / Deep Q victim | 未实现 | 需要模型结构、训练循环、checkpoint、评估脚本。 |
| Neural-pruned MCTS / PUCT | 未实现 | 需要 policy/value prior、PUCT search、与普通 MCTS 的对照实验。 |
| 1000+ rounds 正式实验 | 未完成 | 当前只有轻量 tournament/ablation 脚本；需要固定 seeds、参数表、正式 CSV 和汇总。 |
| 论文/报告级结果表 | 未完成 | 需要把 tournament 输出整理成 win rate、Elo、trap_events、path_delta 等表格。 |
| Web UI 离线资源 | 部分完成 | pawn 模型已本地化；Three.js 仍从 CDN 加载，完全离线展示需要 vendored JS 或构建流程。 |

## 当前验证状态

最近一次验证命令：

```powershell
node --check quoridor\web\static\app.js
python -m unittest discover -s tests -v
```

验证结果：

- JavaScript 语法检查通过。
- Python 单元测试通过，当前测试数为 40。
- Web UI 已在本地 `http://127.0.0.1:8766` 做过桌面和移动视口 smoke check。

## 算法审计记录

2026-06-25 先审计现有算法，再进入新算法设计。当前结论：

- 核心规则、训练接口、Web session、Greedy BFS、Minimax、MCTS、PathLure、DepthTrap、RolloutPoison 的单元测试全部通过。
- `experiments/run_tournament.py --preset smoke --games-per-pair 1 --max-turns 30` 已通过并写出 CSV。
- `experiments/run_tournament.py --preset adversarial --games-per-pair 1 --max-turns 5` 已通过并写出 CSV，但全量 adversarial preset 比较慢，后续正式实验前需要单独做性能预算和参数表。
- 修正了 evaluation 中的 `trap_events` 统计语义：现在只统计对手从非陷阱状态进入陷阱状态的转移，不再在陷阱持续存在时每回合重复计数。

本轮新增验证命令：

```powershell
python -m compileall -q quoridor tests
python -m unittest discover -s tests -v
python experiments\run_tournament.py --preset smoke --games-per-pair 1 --max-turns 30 --output tmp\algorithm_audit_tournament_smoke_after.csv
python experiments\run_tournament.py --preset adversarial --games-per-pair 1 --max-turns 5 --output tmp\algorithm_audit_tournament_adversarial_tiny.csv
```

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

## 交接备注

- 新 agent 只需要实现 `choose_action(state, legal_actions)`。
- 规则引擎已经比较完整，除非发现规则 bug，不建议在实验阶段改 `quoridor/core/`。
- 搜索类 agent 的性能关键在候选墙位筛选，不要在每一步完整枚举所有墙再深搜。
- Web UI 是展示和调试工具，不是训练入口；训练接口以 `DiscreteQuoridorEnv` 为准。
