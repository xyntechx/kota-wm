# Sample efficiency: Dyna (`micro-horizon8`) vs model-free baselines

**Question.** How many real-environment samples does the current default, `micro-horizon8`, use, and how many real samples does a policy trained without a world model need to reach the same return?

**Answer.** `micro-horizon8` reaches 254.3 using **68,224 real steps**, plus about 1.01M imagined steps. The model-free baselines need far more real steps to reach comparable returns: about **70×** for DQN to reach 200, and **190–330×** for PPO to reach 250:

- **PPO with a GRU over the same 64-step window:** reached 250 after 10–12M real steps, on 2 of 3 seeds.
- **Memoryless PPO:** reached 250 after 17.5M real steps, on 1 of 3 seeds.
- **DQN with replay:** the most sample-efficient model-free method. It reached 200 after 1.25–1.5M real steps, but plateaued around 215.

Counting imagined steps as samples too, Dyna is still about 5× (vs DQN) to 31× (vs PPO GRU) more efficient to 200 return.

All numbers use the same evaluation: 200 greedy episodes (argmax action), seed 0, capped at 256 steps. Unless noted, results are 3 seeds (3/4/5).

## Headline

| Method | Real steps to first ≥ 200 | Real steps to first ≥ 250 | Final return (mean ± SE) | Completion | Real steps used |
|---|---|---|---|---|---|
| **Dyna `micro-horizon8`** | **17k / 20k / 20k** | **52k / 65k / 43k** | **254.3 ± 0.5** | **0.975** | **68k** |
| DQN + replay, memoryless MLP | 1.50M / 1.25M / 1.25M | never | 216.5 ± 2.8 | 0.886 | 3M |
| DQN + replay, GRU-64 | never | never | 61.5 ± 1.2 | 0.600 | 1M (stopped early) |
| PPO, GRU-64, ε = 0.3 + warmup | — / 5.5M / 11.5M | — / 10.0M / 12.0M | 216.6 ± 41.4 (134 / 258 / 258) | 0.883 | 20M |
| PPO, memoryless MLP, ε = 0.3 + warmup | 4.5M / 2.0M / 7.0M | — / 17.5M / — | 244.3 ± 4.7 | 0.969 | 20M |

Ratios of mean real steps to reach 200, relative to Dyna's 19k:

| Method | Real steps to 200 | vs Dyna |
|---|---|---|
| DQN, memoryless | 1.33M | about 70× |
| PPO, memoryless | 4.5M | about 235× |
| PPO, GRU (2 seeds) | 8.5M | about 440× |

Only the PPO GRU (2 of 3 seeds) matched or beat Dyna's final return, finishing at 257.5 and 258.3.

## What `micro-horizon8` consumes

Per 2,100-iteration run (`abl2/`, `abl2-s4/`, `abl2-s5/` on volume `kota-wm-checkpoints`):

| | Seed 3 | Seed 4 | Seed 5 |
|---|---|---|---|
| Real env steps | 68,224 | 68,224 | 68,224 |
| Imagined steps | 1,014,938 | 1,008,230 | 1,007,705 |
| Real + imagined | 1,083,162 | 1,076,454 | 1,075,929 |

**Where the real steps come from.** They are 1,024 warmup steps plus 32 per iteration, and every real step counts, including exploration steps. The collector's ε = 0.3 draws on `exploration_action`: uniform over graph-legal moves and no-op, with a 10% chance of a fully random action.

**Where the imagined steps come from.** Each iteration imagines up to 64 rollouts × 8 steps = 512 steps, and fewer when a rollout is predicted to terminate early.

**How often data is reused.**
- PPO trains 4 epochs on each imagined batch.
- The world model trains on a 10,000-step replay buffer: 30 updates of 64 windows of up to 64 steps each, every iteration. A transition stays in the buffer for about 310 iterations, so each real step is revisited very many times.

**Combined real + imagined steps at each return threshold:**

| Return | Seed 3 | Seed 4 | Seed 5 |
|---|---|---|---|
| ≥ 100 | 136k | 183k | 181k |
| ≥ 200 | 241k | 288k | 287k |
| ≥ 230 | 451k | 393k | 444k |
| ≥ 250 | 819k | 1.02M | 654k |

## Model-free setups

Code: `model_free.py` (PPO) and `dqn.py` (DQN), launched through `modal_train.py::model_free_sweep`. Specs: `sweeps/model_free.json` (`mf1`), `sweeps/model_free_warmup.json` (`mf2`) and `sweeps/dqn.json` (`dqn1`).

**Policy input.** Each step is represented as [previous action, 5 observation tokens]. The window policies see the last 64 steps, the same window as the world model's `context_steps`. Steps before the episode start are filled with `<PAD>`.

**Encoders:**

| Encoder | Architecture | Parameters |
|---|---|---|
| GRU-64 | 32-dim embeddings → Linear(192, 256) → GRU(256) over the 64 steps → MLP with 1024 hidden units | 1.77M |
| MLP-64 | Flattened embeddings of all 64 × 6 tokens (12,288 dims) → MLP with 1024 hidden units | 13.6M |
| Memoryless MLP | The same MLP over the current step only (1 × 6 tokens) | 1.26M |

For comparison, Dyna's world model has 1.61M parameters and its policy 1.32M.

**Exploration, shared with Dyna.** Variants marked "ε = 0.3" act from the mixture 0.7·π + 0.3·q, where q is exactly `dyna.exploration_action`'s distribution. PPO trains on the mixture's likelihoods, so the update stays on-policy. "Warmup" adds Dyna's 1,024 steps of pure exploration at the start. Evaluation is greedy on π alone, as in Dyna.

**PPO settings.** 64 environments × 128 steps = 8,192 steps per update, 4 epochs, minibatch 1,024, learning rate 1e-4. γ = 0.995 and λ = 0.95 match Dyna. The rest: clip 0.2, value clip 0.2, entropy coefficient 0.02, rewards divided by 20 (Dyna's `reward_scale`), and a bootstrap at the 256-step time limit. Each real step is used 4 times, then discarded.

**DQN settings.**
- **Learning rule:** Double DQN with a dueling head on the same encoders, 3-step returns, γ = 0.995, Huber loss, learning rate 1e-4, and a hard target-network update every 2,000 gradient steps.
- **Replay:** a 1M-step buffer. It stores one step per slot and rebuilds each window when sampled, never across episode boundaries.
- **Update ratio:** 16 environments step once, then 4 updates of batch 256, so each real step is replayed about 64 times.
- **Exploration:** Dyna's warmup of 1,024 steps and ε = 0.3.
- **Checks:** the buffer matched a slow step-by-step reference on 180 sampled n-step targets and windows, covering ring wraparound and time-limit cutoffs. A separate check confirmed the replayed windows match the acting windows (16,000 of 16,000).

**Hardware.** All model-free runs used one L4 GPU with 4 CPUs on Modal. The Dyna runs used an L40S.

## Return at fixed real-step budgets

Return at the last evaluation at or before each budget, per seed:

| Method | 68k (Dyna's full budget) | 250k | 1M | Final |
|---|---|---|---|---|
| **Dyna `micro-horizon8`** | **255 / 254 / 254** | — | — | 255 / 254 / 254 (68k) |
| DQN, memoryless | 126 / 122 / 101 | 143 / 148 / 150 | 200 / 196 / 199 | 217 / 221 / 211 (3M) |
| DQN, GRU-64 | 78 / 92 / 77 | 73 / 87 / 84 | 59 / 63 / 62 | 59 / 63 / 62 (1M) |
| PPO, memoryless + warmup | −20 / −20 / −20 | −18 / −14 / −18 | 83 / 115 / 20 | 236 / 253 / 244 (20M) |
| PPO, GRU-64 + warmup | −56 / −24 / −38 | −19 / −19 / −20 | 12 / 12 / −16 | 134 / 258 / 258 (20M) |

For reference, random flow-following exploration scores about −57, and crashing immediately scores −20. At Dyna's budget, every PPO run is still at one of those two levels. DQN is the only model-free method with any real return by then.

## Wall-clock time

| Method | Real steps/s | Wall time to first ≥ 200 | Wall time to first ≥ 250 | Total run |
|---|---|---|---|---|
| Dyna `micro-horizon8` (L40S) | 26–37, plus 386–542 imagined/s | 10 / 9 / 12 min | 33 / 30 / 26 min | 0.52–0.73 h |
| DQN, memoryless (L4) | 580–820 | 43 / 26 / 35 min | never | 1.0–1.4 h (3M) |
| DQN, GRU-64 (L4) | ~275 | never | never | ~1.0 h (1M) |
| PPO, memoryless + warmup (L4) | 6,700–7,500 | 12 / 6 / 18 min | — / 38 min / — | 0.73–0.77 h (20M) |
| PPO, GRU-64 + warmup (L4) | 3,200–5,100 | — / 24 / 52 min | — / 42 / 53 min | 1.08–1.41 h (20M) |

PPO processes about 200× more real steps per second than Dyna. Because the city simulator is essentially free (about 5 µs per step), that roughly cancels PPO's sample inefficiency: both reach 200 in about 6–18 minutes. DQN spends its compute on replayed updates, so it is the slowest on wall-clock. With an expensive or real-world environment, wall time would track sample count, and Dyna would win by roughly the sample-efficiency factors above.

Caveats on timing:
- Different GPUs: L40S for Dyna, L4 for the model-free runs.
- The model-free runs are limited by stepping the Python environment on the CPU.
- Dyna's elapsed times exclude its snapshot evaluations, which ran in separate containers. The model-free times include their own evaluations.

## Findings

1. **Plain PPO gets stuck in a "crash immediately" local optimum.** Leaving the road costs −20 once and ends the episode. Wandering costs −20 for each of the 10 tasks it fails. Without the exploration mixture, PPO's entropy collapsed within about 200k steps, and the policy drove off-road on its first step. With ε = 0.3, the GRU got much further (108–145 at 2.5M steps vs 15–37 without). The MLP-64 did not.

2. **Adding Dyna's warmup made no clear difference** to PPO's steps-to-200 (`mf2` vs `mf1`). With a 1,024-step warmup against 8,192-step updates, that is expected.

3. **Memory matters less than it looks, and a long window is hard to learn from reward alone.**
   - **The memoryless ceiling on expert routes.** "Turn … on 3rd Street" tasks need only the current position. "Turn … in n units" tasks count from the task's start, but each task starts one cell past the intersection of the previous turn, so position usually implies the start. On expert trajectories (always follow directions, never wait), the best memoryless policy completes 94.7% of tasks: 100% of road-name tasks and 89.4% of "units" tasks.
   - **The learned memoryless policy beats that ceiling.** Memoryless PPO seed 4 reached 97.7% completion, 93.8% on "units" tasks. Randomizing the traffic-light token in its input cost 10 points on "units" tasks but only 4 on road-name tasks. It also takes about 1.2 no-ops per task. Together these suggest it uses the light cycle, or its own waiting, as external memory. That mechanism is not proven.
   - **Learning curves.** Memoryless policies learn fastest (PPO and DQN alike). The GRU-64 learns slowly and sharply: seeds 4 and 5 jumped by about 100 return within 1–2M steps once they began using memory. Only the GRU reached Dyna-level completion (98.5–98.9%). The flattened MLP-64 (13.6M parameters) was the worst variant.

4. **DQN with replay is the most sample-efficient model-free method, but it plateaus.**
   - The memoryless DQN reaches 100 in 66k real steps and 200 in 1.25–1.5M, then plateaus at about 215.
   - The GRU-64 DQN peaks early (85–92 at 66–131k steps), then declines steadily to about 60 by 1M steps, with a flat loss, consistently across all 3 seeds.
   - The window check above rules out a data bug. This looks like a known weakness of Q-learning with a recurrent encoder over long windows, and these settings were not tuned.

5. **Seed variance is much larger for model-free runs.** PPO GRU-64 finished at 134 / 258 / 258. Dyna finished at 255 / 254 / 254.

## Caveats

- **Tuning effort is unequal.** Dyna's configuration came out of two ablation rounds. The model-free baselines use standard settings with one round of fixes: the exploration mixture, a lower learning rate and entropy 0.02 for PPO, and none for DQN. Better-tuned model-free agents would likely close some of the gap, especially DQN and recurrent DQN.
- **Incomplete runs.** The seed-5 runs of `mf1` never started (queued on GPUs, then the sweep was stopped), and the other `mf1` runs were stopped at 2.5–5M steps. The DQN GRU runs were stopped at 1M of a 3M budget, by decision, to save time.
- **What counts as a sample.** Real steps count every environment step used for training, including exploration. Evaluation episodes are not counted, for Dyna or the model-free runs.

## Files and data

- **Dyna metrics:** `abl2/micro-horizon8`, `abl2-s4/micro-horizon8`, `abl2-s5/micro-horizon8` on volume `kota-wm-checkpoints` (`metrics.json`), plus `sweeps/results/abl2_evals.json` (snapshot evaluations every 100 iterations).
- **Model-free runs:** `mf1/<variant>-s<seed>`, `mf2/{gru,obs}-warm-s<seed>` and `dqn1/{gru,obs}-s<seed>` on the same volume. Each holds `metrics.json` (per-update training stats plus evaluations) and the best checkpoint (`policy_best.pt` or `q_best.pt`).
- **Evaluation schedule:** the model-free runs were evaluated at doubling step counts up to 250k or 500k, then every 250k (DQN) or 500k (PPO). Steps-to-threshold values are therefore upper bounds at that resolution: the first evaluation at or above the threshold.
