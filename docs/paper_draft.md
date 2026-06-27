# From Developing to Exploiting Quoridor Agents: Minimal Adversarial Policies Against Search and Learning Baselines

状态：论文草稿骨架，等待正式实验结果填充。

## Working Abstract

Quoridor provides a compact setting for studying how search and learning agents fail under adversarial wall placement. This project builds a rule-complete Quoridor environment, implements search and reinforcement-learning baselines, and evaluates minimal adversarial policies designed to exploit specific algorithmic blind spots. The central hypothesis is that simple, interpretable trap policies can reliably degrade greedy shortest-path agents, depth-limited minimax, finite-budget MCTS, and deterministic Q-learning victims without requiring a stronger general-purpose Quoridor player.

## Research Question

Can minimal adversarial policies exploit predictable weaknesses in Quoridor agents more efficiently than generic stronger search, and can those weaknesses be categorized by the victim algorithm's decision mechanism?

## Contributions

1. A tested Quoridor environment with legal movement, wall placement, path-preservation checks, discrete RL action encoding, and a browser-based visualization tool.
2. Baseline agents: Random, Greedy BFS, depth-limited Minimax with alpha-beta, finite-budget MCTS, tabular Q-learning, linear Approx-Q, Deep-Q, and heuristic-prior PUCT.
3. Minimal adversarial policies mapped to victim weaknesses: Path-Lure, Depth-Trap, Rollout-Poison, Counterfactual Trap Search, and Argmax-Q Trap.
4. A reproducible tournament harness reporting win rate, Elo-style ratings, wall usage, trap events, path delta, and draw rate.

## Method Overview

### Environment

The game uses the standard 9x9 Quoridor board. Each player starts with 10 walls. A wall action is legal only if it does not overlap or cross existing walls and both players retain at least one path to their goal row. The environment exposes both engine actions and a 209-dimensional fixed action space for RL experiments.

### Baselines

| Baseline | Implementation | Role in study |
| --- | --- | --- |
| Random | `RandomAgent` | Low-skill stochastic control. |
| Greedy BFS | `GreedyBFSAgent` | Shortest-path victim. |
| Minimax | `MinimaxAgent` | Depth-limited adversarial search victim. |
| MCTS | `MCTSAgent` | Finite-budget UCT victim. |
| Tabular Q | `QLearningAgent` | Deterministic argmax RL victim after training. |
| Approx-Q | `ApproxQLearningAgent` | Linear function-approximation RL baseline. |
| Deep-Q | `DeepQAgent` | DQN baseline trained with PyTorch and legal-action masking. |
| PUCT | `PUCTAgent` | Neural-pruned MCTS interface baseline using heuristic priors until a trained network is available. |
| AlphaZero-style PUCT | `AlphaZeroAgent` | Policy/value-network-compatible PUCT wrapper; disabled in Web until a checkpoint exists. |

### Adversarial Policies

| Policy | Target weakness | Current status |
| --- | --- | --- |
| Path-Lure | Greedy BFS ignores path diversity. | Implemented. |
| Depth-Trap | Minimax horizon limitations. | Implemented. |
| Rollout-Poison | MCTS shallow rollout ambiguity. | Implemented. |
| Counterfactual Trap | Robust scoring over plausible victim responses. | Implemented. |
| Argmax-Q Trap | Deterministic RL policy can be response-modeled. | Implemented initial version. |

## Experimental Plan

Primary comparisons:

1. Baseline round-robin: Random, Greedy BFS, Minimax, MCTS, Q-learning, Approx-Q, Deep-Q, PUCT.
2. Adversarial round-robin: each trap policy against its intended victim and non-target baselines.
3. Ablation: vary trap weight, response width, wall candidate budget, and MCTS/PUCT simulation budget.
4. RL sensitivity: compare untrained, lightly trained, and longer-trained Q policies.
5. AlphaZero progression: compare heuristic PUCT, randomly initialized AlphaZero wrapper, bootstrapped policy/value checkpoints, and self-play checkpoints.

Primary metrics:

- Win rate and draw rate.
- Elo-style standing from the tournament harness.
- Average trap events.
- Average opponent path delta after wall actions.
- Average wall actions per game.
- Latency per agent step for Web UI usability.
- Policy improvement strength: score rate of network-only policy, PUCT-guided policy, and prior best checkpoint.

## AlphaZero-Style Final Agent Plan

The final strong-agent direction is AlphaZero-style self-play rather than DQN-only control. The current codebase now exposes the required interfaces:

- `PUCTAgent.search_policy(...)` returns visit-count policy targets.
- `AlphaZeroNet` predicts action logits over the fixed 209-action space plus a scalar value.
- `AlphaZeroAgent` injects neural priors and values into PUCT when a checkpoint is available.
- `alphazero_loss(...)` combines value regression and policy cross entropy.
- `experiments/train_alphazero.py` starts a single self-play training run and writes `experiments/results/alphazero_policy_value.pt`.
- `experiments/run_alphazero_config.py` runs staged, chunked, resumable long training from `experiments/configs/alphazero_remote_long.json`.
- `AlphaZeroBatchEvaluator` batches policy/value network calls for PUCT leaf expansion and avoids duplicate prior/value inference for the same state.
- `experiments/run_batched_selfplay.py` generates self-play shards with arena-style workers and can train on the merged examples with large minibatches.

Long-run training loop:

1. Run self-play with current best checkpoint or heuristic fallback.
2. Store `(flat_observation, visit_count_policy, game_outcome)` examples.
3. Train `AlphaZeroNet` on replayed self-play examples.
4. Evaluate candidate checkpoint in the arena against current best and enhanced heuristic baselines.
5. Promote only if the candidate passes a fixed score-rate threshold.

The current implementation now separates three engineering concerns that should be reported distinctly in the paper: teacher bootstrap data, batched MCTS self-play data generation, and arena gating. This is important because early experiments showed that naive serial self-play mostly produced max-turn draws and left GPU inference underutilized. The updated path uses batched PUCT leaf evaluation plus parallel self-play shards before large-batch supervised/RL updates.

An initial smoke checkpoint has been produced from 2 self-play games with 60 examples and 2 optimizer updates on CUDA. This only validates the training/checkpoint path; it is not a claim of strong play. The remote long-run config intentionally starts from scratch with a clean 256-hidden policy/value network.

This follows the high-level pattern from AlphaGo Zero and AlphaZero, but the implementation must be scaled down for local GPU constraints. The paper should distinguish clearly between:

- heuristic-enhanced PUCT baseline,
- AlphaZero-compatible interface,
- actually trained AlphaZero-style checkpoint.

## Reproducibility Commands

```powershell
python -m unittest discover -s tests -v
python experiments\train_q_learning.py --episodes 500 --max-turns 120 --output experiments\results\q_learning_policy.json
python experiments\train_approx_q.py --episodes 500 --max-turns 120 --output experiments\results\approx_q_policy.json
F:\Programs\PythonEnv\torch10\python.exe experiments\train_deep_q.py --episodes 500 --max-turns 120 --device cuda --output experiments\results\deep_q_policy.pt
python experiments\run_alphazero_config.py --config experiments\configs\alphazero_remote_long.json --dry-run
python experiments\run_alphazero_config.py --config experiments\configs\alphazero_remote_long.json --resume
python experiments\run_tournament.py --preset research --games-per-pair 10 --max-turns 150 --workers 4 --resume --output experiments\results\tournament_research_games.csv --matrix-output experiments\results\tournament_research_matchups.csv --score-matrix-output experiments\results\tournament_research_scores.csv
```

## Results

Preliminary trap-efficacy results are available from the arena output files. The optimized target run uses per-game seed offsets, records per-game elapsed time, and reports Wilson 95% confidence intervals.

Targeted trap-victim summary:

| Trap | Target | Games | Wins | Losses | Score rate | 95% CI | Avg trap events | Avg target path delta | Preliminary interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| PathLure | Greedy BFS | 20 | 20 | 0 | 1.0000 | [0.8389, 1.0000] | 0.000 | -7.000 | Win-rate supported; current trap-event metric does not support the path-trap mechanism. |
| DepthTrap | Minimax d1 | 20 | 19 | 1 | 0.9500 | [0.7639, 0.9911] | 0.300 | 0.350 | Supported. |
| RolloutPoison | MCTS 5 | 20 | 18 | 2 | 0.9000 | [0.6990, 0.9721] | 0.400 | 0.750 | Supported. |
| CounterTrap | MCTS 5 | 20 | 19 | 1 | 0.9500 | [0.7639, 0.9911] | 0.300 | 2.250 | Supported. |
| ArgmaxQTrap | Q-learning | 20 | 13 | 6 | 0.6750 | [0.4567, 0.8369] | 1.150 | 0.200 | Directional signal; needs more games and/or stronger Q victim training. |

Source files:

- `experiments/results/trap_targets_optimized_games.csv`
- `experiments/results/trap_targets_optimized_matchups.csv`
- `experiments/results/trap_targets_optimized_scores.csv`
- `experiments/results/trap_effectiveness_optimized_targets.csv`

Do not manually invent win rates or Elo values. Any later results section must cite generated CSV files.

Planned tables:

1. Overall tournament standings.
2. Intended-victim matchup matrix.
3. Ablation table for trap parameters.
4. Agent latency table for Web UI settings.
5. AlphaZero checkpoint progression table: stage, games, simulations, arena score rate, and promotion decision.

Arena outputs:

- `tournament_research_games.csv`: one row per game, resume-safe.
- `tournament_research_matchups.csv`: long-format matchup matrix with per-agent-per-opponent metrics.
- `tournament_research_scores.csv`: wide score-rate matrix for heatmaps and paper tables.

## Related Work Notes

Use verified references only:

| Topic | Candidate reference | Verification link |
| --- | --- | --- |
| Q-learning and function approximation | Sutton and Barto, *Reinforcement Learning: An Introduction*, second edition | https://incompleteideas.net/book/the-book-2nd.html |
| UCT | Kocsis and Szepesvari, "Bandit Based Monte-Carlo Planning" | https://link.springer.com/chapter/10.1007/11871842_29 |
| MCTS taxonomy | Browne et al., "A Survey of Monte Carlo Tree Search Methods", DOI `10.1109/TCIAIG.2012.2186810` | https://repository.essex.ac.uk/4117/ |
| Policy/value-guided search | Silver et al., "Mastering the game of Go without human knowledge" | https://discovery.ucl.ac.uk/10045895/ |
| General AlphaZero algorithm | Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" | https://arxiv.org/abs/1712.01815 |
| Heuristic MCTS variants | Gelly and Silver, "Monte-Carlo Tree Search and Rapid Action Value Estimation in Computer Go" | https://www.cs.utexas.edu/~pstone/Courses/394Rspring13/resources/mcrave.pdf |
| Implicit minimax backups in MCTS | Lanctot et al., "Monte Carlo Tree Search with Heuristic Evaluations using Implicit Minimax Backups" | https://mlanctot.info/files/papers/cig14-immcts.pdf |
| Quoridor MCTS | Respall et al., "Monte Carlo Tree Search for Quoridor" | https://www.researchgate.net/publication/327679826_Monte_Carlo_Tree_Search_for_Quoridor |

## Limitations

- Current PUCT uses heuristic priors and heuristic values, not a trained neural policy/value network.
- Current Approx-Q is linear and interpretable; it is not a Deep Q-Network.
- Current Deep-Q training uses GPU for neural updates, but legal action generation and rule checks remain CPU-bound.
- Tournament settings are still small-scale smoke settings until formal runs are generated.
- Web UI latency is measured locally and may differ across machines.

## AI Disclosure Draft

AI assistance was used for code implementation support, debugging, documentation scaffolding, and draft organization. Final claims, experimental results, and references must be verified against the repository outputs and cited sources before submission.
