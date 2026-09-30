# abl2100: ablation report (gpt-mini, 2100 iterations, seeds 3/4/5)

All runs were trained on Modal from scratch to iteration 2100. Each policy was scored in the real `City` environment every 100 iterations. The compact configurations were run with seeds 3, 4 and 5. The grid runs have seed 3 only; the original two were stopped early, and a later grid copy of `micro-horizon8` (`grid2`) ran to 2100. The canonical baseline is `compact-base`: **219.0 ± 13.3 final return (mean ± SE over 3 seeds; per seed 244 / 198 / 216) and 89.7% task completion**.

A second round of ablations (`abl2`) built on the resulting default and is summarized in "Second round (abl2)" below; it made `micro-horizon8` (gpt-micro, imagination horizon 8) the new default. The 3-seed results in the next section supersede the single-seed conclusions. The single-seed (seed 3) analysis follows further down for reference.

## Headline (3 seeds)

| Question | Answer |
|---|---|
| Does anything beat the baseline? | **Yes, `compact-wm-actobs`.** `base` itself is a high draw (see the vocabulary row); against the pooled mean of the 12 baseline-equivalent runs (194.9), `wm-actobs` is even further ahead. Averaged over iterations 1900–2100 it scores 247.7 ± 4.4 vs 217.2 ± 10.9 (+30.5 ± 11.8). Every seed reaches 200 return by iteration 600–800, vs 1500 / never / 1500 for the baseline. Its 3-seed mean passes the baseline's final return (219) at iteration **800**; the baseline's own mean first gets there at 2000. Nothing else beats the baseline. |
| Does it hold with the trimmed vocabulary? | **Yes.** `compact-vocab-wm-actobs` (trimmed vocabulary + `action+obs`) scores 244.4 ± 2.9 final, 243.1 ± 2.7 late (+25.9 ± 11.2 vs base, −4.6 ± 5.2 vs `wm-actobs`). It has the smallest seed spread of any strong config (239–248), and its seeds reach 200 at iterations 500 / 800 / 900. |
| Why does `wm-actobs` help? | Reward and termination gradients reach the observation latents the policy reads. With the same head inputs but the observation mean detached (`wm-actobs-sg`), the run is indistinguishable from the baseline (−5 ± 12 at iterations 500–900, −18 ± 24 late) and far behind `wm-actobs` (−101 ± 10 at iterations 500–900). |
| Policy latent: last vs mean vs both? | The observation-token mean alone matches "both" (`pi-mean`: −6 ± 19). The last hidden state alone fails on every seed (−1.4 final). |
| Mode-specific vocabulary? | **No effect; the code is provably equivalent.** With identical per-token initial weights and a sampler whose random-number use doesn't depend on vocabulary size, the trimmed and shared vocabularies train bit-identically. The raw gap vs `base` (−41 ± 15) is seed noise: trimming changes layer shapes, so with the same seed every weight and every sampled token differs. Among 12 runs that are algorithmically identical to the baseline, `vocab` is a low draw (bottom ~8% of 3-run means) and `base` a high one (top ~10%). See "Vocabulary: equivalence check" below. |
| bf16 autocast? | No measurable effect (`baseline-exact` − `base`: −20 ± 26). |
| Is memory needed? | **Yes.** Shared latents from the current observation alone (`pi-current`) score −92 ± 32 vs the baseline (122 final, all 3 seeds below 145). The environment hides progress within the current task, so the policy needs history. |
| Separating policy and world model? | It hurts. A separate policy transformer over the same context (`pi-seq`) reaches 67 ± 7; one on the latest observation only (`pi-obs`) reaches 167 ± 6. A policy that must learn its own representation from PPO alone is far less sample-efficient than one reading world-model latents. |
| Grid mode vs compact? (seed 3 only) | The original grid runs failed. Their policies drove off-road within 2–4 steps (≈ −20 return), because the grid world model couldn't learn reward or termination (reward accuracy 0.2–0.4); they were stopped at iterations 1000 and 900. A later grid copy of `micro-horizon8` (`grid2`, see "Grid mode with the micro-horizon8 settings") does learn: flat until about iteration 700, then 86.5 at 2100 and still rising, against compact's 255. |

## Multi-seed results (seeds 3, 4, 5)

Per-seed evaluations are in `sweeps/results/abl2100_seeds_evals.json`. Seed-4 and seed-5 runs are `abl2100-s4/*` and `abl2100-s5/*` on the volume (specs: `sweeps/ablations_s4.json`, `sweeps/ablations_s5.json`). The late average smooths single-eval noise: each seed's evals at 1900, 2000 and 2100 are averaged, then the mean and SE are taken over seeds. `*` marks a difference from the baseline larger than 2 SE.

| Config | Final (2100) mean ± SE | Per seed (3 / 4 / 5) | Late avg mean ± SE | Δ vs base (late) | Completion (2100) | Iteration each seed first reaches 200 |
|---|---|---|---|---|---|---|
| **compact-base** (baseline) | 219.0 ± 13.3 | 244 / 198 / 216 | 217.2 ± 10.9 | — | 0.897 | 1500 / — / 1500 |
| **compact-wm-actobs** | **240.6 ± 7.9** | 243 / 253 / 226 | **247.7 ± 4.4** | **+30.5 ± 11.8 \*** | **0.934** | **800 / 600 / 700** |
| **compact-vocab-wm-actobs** | **244.4 ± 2.9** | 246 / 239 / 248 | **243.1 ± 2.7** | **+25.9 ± 11.2 \*** | **0.943** | **500 / 800 / 900** |
| compact-pi-mean | 217.5 ± 15.3 | 244 / 191 / 217 | 211.6 ± 15.7 | −5.6 ± 19.1 | 0.895 | 1500 / — / 1800 |
| compact-wm-actobs-sg | 196.2 ± 22.6 | 169 / 241 / 178 | 199.7 ± 21.2 | −17.5 ± 23.8 | 0.843 | — / 1200 / 2000 |
| compact-baseline-exact | 194.9 ± 26.1 | 160 / 246 / 179 | 197.0 ± 24.1 | −20.2 ± 26.4 | 0.841 | — / 1400 / — |
| compact-vocab | 169.1 ± 11.5 | 147 / 184 / 177 | 176.5 ± 9.7 | −40.7 ± 14.6 \* | 0.760 | — / — / 1900 |
| compact-pi-obs | 167.1 ± 6.4 | 159 / 180 / 162 | 174.9 ± 6.7 | −42.3 ± 12.8 \* | 0.768 | — |
| compact-pi-current | 121.7 ± 19.6 | 141 / 141 / 83 | 125.2 ± 30.2 | −92.0 ± 32.1 \* | 0.631 | — |
| compact-pi-seq | 66.7 ± 7.2 | 73 / 75 / 52 | 61.2 ± 5.1 | −156.0 ± 12.0 \* | 0.570 | — |
| compact-pi-last | −1.4 ± 1.8 | 2 / −2 / −4 | −2.2 ± 2.2 | −219.4 ± 11.1 \* | 0.634 | — |

Seed-mean real-environment return by iteration:

| Config | 300 | 500 | 700 | 900 | 1200 | 1500 | 1800 | 2100 |
|---|---|---|---|---|---|---|---|---|
| compact-base | 4 | 61 | 109 | 139 | 146 | 197 | 198 | 219 |
| **compact-wm-actobs** | **100** | **165** | **209** | **226** | **242** | **245** | **235** | **241** |
| **compact-vocab-wm-actobs** | **92** | **176** | **173** | **216** | **233** | **244** | **246** | **244** |
| compact-pi-mean | 23 | 63 | 104 | 129 | 148 | 182 | 208 | 217 |
| compact-baseline-exact | 4 | 62 | 98 | 123 | 138 | 168 | 181 | 195 |
| compact-wm-actobs-sg | 5 | 76 | 99 | 131 | 161 | 164 | 186 | 196 |
| compact-vocab | −10 | 37 | 88 | 99 | 136 | 145 | 175 | 169 |
| compact-pi-obs | 59 | 81 | 75 | 117 | 119 | 151 | 170 | 167 |
| compact-pi-current | −17 | 12 | 27 | 41 | 59 | 87 | 112 | 122 |
| compact-pi-seq | −11 | −2 | 2 | 18 | 32 | 49 | 58 | 67 |
| compact-pi-last | −22 | −19 | −18 | −18 | −13 | −11 | −8 | −1 |

Key pairwise comparisons (mean ± SE over seeds):

| Comparison | Iterations 500–900 avg | Iterations 1900–2100 avg |
|---|---|---|
| wm-actobs − wm-actobs-sg | +100.6 ± 9.8 | +48.0 ± 21.6 |
| vocab-wm-actobs − wm-actobs (differ only in vocabulary) | — | −4.6 ± 5.2 |
| wm-actobs-sg − base | −5.4 ± 11.5 | −17.5 ± 23.8 |
| vocab − base (differ only in vocabulary) | −31.9 ± 10.3 | −40.7 ± 14.6 |
| baseline-exact − base (differ only in autocast) | −15.4 ± 11.3 | −20.2 ± 26.4 |
| pi-current − base (differ only in history) | −80.7 ± 38.4 | −92.0 ± 32.1 |
| pi-obs − pi-current | +66.4 ± 38.6 | +49.7 ± 30.9 |

What changed from the single-seed reading:
- **Memory:** it went from "not settled" to clearly needed. On seed 3, `base` looked like a high draw and `pi-current` seemed within its spread. Seeds 4 and 5 put `pi-current` far below `base`: it stays under 60 until iteration 1200.
- **Vocabulary:** the raw 3-seed gap looked like a deficit (−41 ± 15). The code check below shows the two vocabularies are computationally identical, so that gap is seed noise and not an effect of trimming.
- **`wm-actobs`:** its lead holds and is the most consistent result in the sweep. It has the lowest seed variance of the strong runs (final SE 7.9), and its seeds reach 200 about twice as fast as the baseline's.
- **Memoryless variants:** the separate observation-only network (`pi-obs`, 167) edges out the world-model latents of an isolated observation (`pi-current`, 122) (+50 ± 31, 1.6 SE, not significant). The world model rarely encodes an observation with no preceding context, which may make those latents less useful.

### Vocabulary: equivalence check

Every token id in the code is looked up by string (`VOCAB[...]`), and no id is hard-coded. To confirm there's no hidden dependency, the same short training run (compact, gpt-nano, 5 Dyna iterations) was done under both vocabularies, with:
- per-token initial weights, keyed by token string and identical in both,
- the 5 unused logits given a −1e4 bias,
- the same seed after initialization.

Results:
- **With `torch.multinomial` sampling:** bootstrap (collection, replay, 20 world-model updates) is identical, but runs diverge at the first imagination step. `multinomial` pairs random draws with vocabulary entries, so a different vocabulary size gives different samples from the same distribution.
- **With a sampler whose random-number use doesn't depend on vocabulary size** (inverse CDF, one uniform per row): all metrics match exactly (relative difference 0.0) through bootstrap and all 5 iterations.

So trimming the vocabulary changes only random draws: initialization (different layer shapes) and imagination samples. Final returns of the 12 runs that are algorithmically identical to the baseline up to randomness (`base`, `baseline-exact`, `vocab`, and `wm-actobs-sg`, whose heads differ only in input) span 147–246 with sd 34. That spread is the noise floor for single configs at 3 seeds. The check script is `vocab_equiv.py` in the session scratchpad; it could be added as a unit test.

### Trimmed vocabulary + `action+obs` (`compact-vocab-wm-actobs`)

This config combines the mode-specific (89-token) vocabulary with `outcome_features="action+obs"`. It was run with seeds 3, 4 and 5; the runs are `abl2100/`, `abl2100-s4/` and `abl2100-s5/` `compact-vocab-wm-actobs`. Its per-seed evaluations were added to `sweeps/results/abl2100_seeds_evals.json`.

- **It keeps the full `wm-actobs` advantage.** Late return 243.1 ± 2.7 vs 247.7 ± 4.4 for `wm-actobs` (−4.6 ± 5.2), both well above `base` (217.2 ± 10.9).
- **Most consistent strong config:** final returns 246 / 239 / 248, with 94.3% completion on average.
- **Consistent with the equivalence check.** The trimmed vocabulary with baseline heads (`compact-vocab`) landed low (147–184). With `action+obs` heads it lands at 239–248, so the earlier low `compact-vocab` numbers came from seed draws, not from the vocabulary.
- **Best snapshot:** seed 5, iteration 1800, **259.1 ± 2.2 return, 97.6% completion**.

### Website model

- **Current (installed in the Website repo, not yet committed or deployed):** the best `micro-horizon8` snapshot (seed 5, iteration 1900: 262.2 ± 1.9, 98.6% completion) is exported to `Website/public/models/kota-wm.onnx` (11.7 MB, down from 21.2 MB). `app/_kota/model.ts` now expects 4 layers and 4 heads (gpt-micro), and its cache name was bumped to `kota-v6`. The 89-token vocabulary is unchanged. `export_onnx.py --verify` matched PyTorch to within 7e-5 over a 600-step rollout with 9 context crops.
- **Previous site model (committed as "new best model with compact vocab"):** the best `compact-vocab-wm-actobs` snapshot (seed 5, iteration 1800: 259.1 ± 2.2). It needs 6 layers and 6 heads.
- **Backups** in `kota-wm/checkpoints/website-backup/`:
  - `kota-wm.2026-09-24.onnx`: the original site model, 177.0 on this evaluator
  - `kota-wm.wm-actobs-s4-iter2000.onnx`: the best shared-vocabulary `wm-actobs` snapshot, 260.0. It needs the 94-token list and 6 layers/heads.
  - `kota-wm.vocab-wm-actobs-s5-iter1800.onnx`: the previous site model above.

## Second round (abl2)

Every `abl2` run starts from the `compact-vocab-wm-actobs` configuration (the default at the time) and changes one thing. The spec is `sweeps/abl2.json`, with every setting pinned, and the per-snapshot data is in `sweeps/results/abl2_evals.json`. Each config was run with seed 3 only, except `micro-horizon8`, which has seeds 3, 4 and 5 (`sweeps/abl2_s4.json`, `sweeps/abl2_s5.json`). For reference, the base config's late average (iterations 1900–2100) was 238 / 244 / 248 over its three seeds, so a single-seed result is notable roughly outside 233–253.

New options added for this round (all off by default): `outcome_loss_weight` (weight on the reward and termination losses), `return_head` (predicts the discounted return from the outcome-head input; training only), `outcome_features="action+obs-all"` (the heads see all 5 observation hidden states instead of their mean), and `KOTA_ORACLE=1` (compact observations also carry the hidden task progress, `City._dir_idx`, as a `progress k` token).

| Run (seed 3) | Changes | Late avg | Final (2100) | Completion | First ≥ 200 | s/iter |
|---|---|---|---|---|---|---|
| base (3 seeds) | — | 238–248 | 246 (seed 3) | 0.957 | 500 / 800 / 900 | 2.1 |
| **oracle** | task progress in the observation | **260** | **264** | **0.987** | 700 | 2.2 |
| gpt-micro | 4 layers, 128 dims | 248 | 250 | 0.936 | 600 | **1.3** |
| horizon8 | imagination horizon 8 | 247 | 251 | 0.979 | 900 | 1.6 |
| obs-all | heads see all 5 observation states | 247 | 250 | 0.957 | 700 | 2.1 |
| wupd15 | 15 world updates per iteration | 246 | 251 | 0.961 | 1000 | 1.7 |
| loss-w3 | reward/termination loss ×3 | 250 | 236 | 0.923 | 600 | 2.1 |
| horizon32 | imagination horizon 32 | 244 | 245 | 0.934 | 900 | 3.0 |
| loss-w10 | reward/termination loss ×10 | 240 | 245 | 0.949 | 600 | 2.1 |
| wupd60 | 60 world updates per iteration | 237 | 249 | 0.960 | 700 | 2.8 |
| ctx32 | 32 context steps | 236 | 226 | 0.918 | 800 | 2.1 |
| return-head | return-prediction head | 228 | 243 | 0.948 | 1000 | 2.2 |
| ctx16 | 16 context steps | 221 | 226 | 0.901 | 1100 | 2.1 |
| gopher-44m | 8 layers, 512 dims | 213 | 181 | 0.866 | 1400 | 3.4 |
| ctx8 | 8 context steps | 205 | 220 | 0.885 | 1500 | 2.0 |

Findings (single seed each):
- **Memory length matters, and the model's memory is imperfect.** Return rises with context length (205 at 8 steps, 221 at 16, 236 at 32, 238–248 at 64). The oracle, which sees task progress directly, beats every base seed (260, 98.7% completion), so about 15 points are still lost to imperfect memory. The oracle changes the environment's observation, so it isn't deployable, but better memory is the most promising direction.
- **Smaller is as good; bigger is worse.** `gpt-micro` matches gpt-mini at about 40% less time per iteration. `gopher-44m` learns slowly and is unstable, and would probably need a different learning rate or longer training.
- **Shaping the latents further doesn't help.** Heavier loss weights and the all-5-states heads match the base; the return head slows early learning. `action+obs` already captures the benefit.
- **Schedule changes cost speed, not results.** Horizons 8 and 32 and 15 or 60 world updates all reach the base score. Horizon 8 is ~25% faster per iteration; 15 world updates is faster but starts slower.

**Why a shorter imagination horizon doesn't hurt.** Results are insensitive to horizon between 8 and 32, for three reasons:
1. The critic covers everything past the horizon. Imagined returns are λ-returns that bootstrap from the value estimate at the last imagined step; with γ = 0.995 and λ = 0.95 the effective credit-assignment window is about 1/(1 − γλ) ≈ 18 steps, and later rewards are heavily discounted.
2. Imagined rollouts drift from reality. Each rollout starts from a real replay state, then the model generates its own observations, and errors compound. Short rollouts stay near real states, where the model is most accurate, which offsets having half as many imagined samples per update.
3. Credit assignment in this task is short-range: rewards are dense and tasks last at most 16 steps.

### micro-horizon8 (3 seeds): the new default

Combining `gpt-micro` with horizon 8:

| Config (3 seeds) | Final (mean ± SE) | Per seed | Late avg | Completion | First ≥ 200 per seed | Time per run |
|---|---|---|---|---|---|---|
| **micro-horizon8** | **254.3 ± 0.5** | 255 / 254 / 254 | 247.5 ± 4.8 | **0.975** | **500 / 600 / 600** | **0.5–0.7 h** (0.9–1.25 s/it) |
| base (compact-vocab-wm-actobs) | 244.4 ± 2.9 | 246 / 239 / 248 | 243.1 ± 2.7 | 0.943 | 500 / 800 / 900 | ~1.2 h (~2.1 s/it) |

| Seed-mean return | 300 | 500 | 700 | 900 | 1200 | 1500 | 1800 | 2100 |
|---|---|---|---|---|---|---|---|---|
| micro-horizon8 | 107 | 179 | 217 | 234 | 238 | 240 | 251 | 254 |
| base | 92 | 176 | 173 | 216 | 233 | 244 | 246 | 244 |

It is at least as good as the base: the late average is statistically the same (+4.4 ± 5.6), and the final checkpoints are higher and very consistent (254–255), though that's one evaluation per seed. It learns faster through mid-training (217 vs 173 at iteration 700), and takes about 40–60% less wall time. Most of the speedup comes from `gpt-micro` (1.3 s/it alone vs 1.25 combined), because a model this small is limited by per-step overhead, so the shorter horizon saves little extra. Its best snapshot (seed 5, iteration 1900: 262.2 ± 1.9, 98.6% completion) is the site model.

### Grid mode with the micro-horizon8 settings (grid2, seed 3)

`grid2/micro-horizon8-grid` copies `abl2/micro-horizon8` exactly and switches only `obs_mode` to `"grid"`. That means gpt-micro, imagination horizon 8, `outcome_features="action+obs"`, the mode-specific vocabulary (64 tokens), 64 context steps, world batch 64, 64 imagined rollouts and seed 3. It ran on an H100 to iteration 2100. The spec is `sweeps/grid2.json` and the per-snapshot data is in `sweeps/results/grid2_evals.json`.

**It is the first grid run that learns.** It stayed near the off-road penalty through iteration 700, like the earlier grid runs, and then climbed without a break to **86.5 ± 3.2 at iteration 2100 (72.8% completion)**. It was still rising steeply at the end: +28 over the last 100 iterations.

| Iteration | 500 | 700 | 900 | 1000 | 1200 | 1400 | 1600 | 1800 | 1900 | 2000 | 2100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Return | −25.7 | −21.7 | −16.1 | −13.2 | −1.2 | 6.8 | 30.8 | 37.1 | 50.5 | 58.5 | **86.5** |
| Completion | 0.08 | 0.24 | 0.56 | 0.55 | 0.64 | 0.65 | 0.71 | 0.68 | 0.68 | 0.68 | 0.73 |
| Episode length | 5.7 | 2.7 | 5.8 | 8.4 | 13.9 | 17.9 | 27.0 | 34.8 | 41.6 | 46.1 | 48.1 |
| Reward MSE | 138 | 135 | 130 | 91 | 90 | 77 | 57 | 61 | 36 | 37 | 31 |
| Termination accuracy | 0.883 | 0.852 | 0.863 | 0.902 | 0.922 | 0.957 | 0.980 | 0.965 | 0.980 | 0.977 | 0.977 |

Evaluation SE is 0.7–3.6 throughout. Compared with the earlier grid runs and with compact mode:

| Run | Return at 900 / 1000 | Return at 2100 | Reward MSE at 1000 | Time per iteration |
|---|---|---|---|---|
| grid-base (gpt-mini, horizon 16, `action` outcomes, shared vocab) | −22.1 / −20.2 | stopped at 1000 | 140 | 17.6 s |
| grid-vocab (the same with mode vocab) | −19.8 / — | stopped at 900 | 164 (at 900) | 19.6 s |
| **grid2 micro-horizon8-grid** | **−16.1 / −13.2** | **86.5** | **91** | **12.4 s** (7.2 h total) |
| compact micro-horizon8 (seed 3) | 231 / 228 | 255 | 3.2 | 1.25 s |

Findings (single seed):
- **The world model learns the grid, slowly.** Reward MSE fell from about 135 to 31, and termination accuracy rose from 0.85 to 0.98, mostly after iteration 1000. The policy's gains track that improvement, which fits the earlier diagnosis: the bottleneck is the world model learning to locate the player among 192 cells to predict outcomes. Token accuracy was 0.99 from the start, because it mostly reflects the static road layout.
- **Return grows mainly through longer, safer episodes.** Episode length went from 8 to 48 steps between iterations 1000 and 2100, while completion rose only from 0.55 to 0.73. The policy first learned to stay on the road, and is only starting to follow instructions reliably.
- **It is still far behind compact mode.** Grid at 2100 (86.5) is where compact `micro-horizon8` was between iterations 200 (69) and 300 (123), on the same 68,224 real steps. Grid imagined fewer steps (700k vs 1.01M), because more rollouts were predicted to end early. It took about 10× longer per iteration.
- **Why this run learned and the old ones didn't is untested.** It differs from `grid-base` and `grid-vocab` in three ways: `action+obs` outcome features (the heads also see the mean over all 194 observation hidden states), gpt-micro instead of gpt-mini, and horizon 8 instead of 16. The old runs were also stopped at 900–1000, before this one's main climb, but at that point they were flat at −20 while this one was already improving (−13 at 1000). Single-seed runs spread widely, so a second seed would be needed before attributing the difference.
- **It has not plateaued.** Its checkpoint and `train_state.pkl` allow an exact resume past 2100, at about 12–17 s per iteration on an H100.

## Single-seed analysis (seed 3)

The rest of this report is the original seed-3 analysis. Where it disagrees with the section above (memory, vocabulary), the 3-seed results take precedence.

## Setup

- **Shared configuration** (from `runs/fullep-v1-mini`): compact observations, `gpt-mini` backbone (6 layers, 6 heads, 192 dims), world-model and agent hidden size 1024, `context_steps` 64, `world_batch_size` 64, `imagined_rollouts` 64, imagination horizon 16, seed 3, trained to iteration 2100. Every run has `autocast: true` (bf16 world model) except `compact-baseline-exact`, which copies the old config verbatim. The spec is in `sweeps/ablations.json`.
- **Evaluation** (`modal_train.py::evaluate_remote` → `evaluate.py`): every 100 iterations the checkpoint is copied to `snapshots/iter_NNNN.pt` and scored in a separate container, in parallel with training. Scoring is 200 greedy episodes in the real environment, seed 0 (the same start cities for every run), with real rewards and at most 256 steps per episode. The world model only encodes real observations into the policy's inputs; it never simulates during scoring. Completion rate is tasks completed / (completed + deviated). A road-following random policy scores −56.8 on the same episodes.
- **"Beats baseline"** means return ≥ 243.5. A **clear beat** means the gap exceeds 2 standard errors of the difference.
- The old `runs/fullep-v1-mini` checkpoint scores 177.0 ± 4.3 (iteration 2100) and 201.5 ± 3.4 (its best, iteration 2040) under the same evaluator. It was trained in many resumed segments and is kept only as a secondary reference.

## Runs

| Run | What changes vs `compact-base` |
|---|---|
| compact-base | none (baseline): policy reads world-model latents at the current observation, `concat(last hidden, mean of observation-token hiddens)` ("both") |
| compact-baseline-exact | `autocast: false` (the old config verbatim) |
| compact-vocab | mode-specific vocabulary: the 5 grid-cell tokens are removed (94 → 89 tokens) |
| compact-pi-last | policy reads only the last hidden state |
| compact-pi-mean | policy reads only the observation-token mean |
| compact-pi-current | policy reads "both", but computed from the world model encoding **only the current observation** (no history) |
| compact-wm-actobs | world-model reward/termination heads read `concat(action-token hidden, observation-token mean)` instead of the action-token hidden alone |
| compact-wm-actobs-sg | as `wm-actobs`, but the observation mean is detached (stop-gradient) |
| compact-pi-obs | separate actor-critic (its own gpt-mini transformer on tokens), latest observation only |
| compact-pi-seq | separate actor-critic, same 64-step token context as the world model |
| grid-base | grid observations (194 tokens per step), shared vocabulary |
| grid-vocab | grid observations, mode-specific vocabulary: row/col/stop tokens removed (94 → 64 tokens) |

## Results against the baseline

| Run | Final return (2100) | Final completion | Best return | First ≥ 243.5 | Clear beat | s/iter |
|---|---|---|---|---|---|---|
| **compact-base** (baseline) | 243.5 ± 4.1 | 0.954 | 243.5 @2100 | — | — | 1.6 |
| **compact-wm-actobs** | 242.9 ± 3.7 | 0.950 | **252.9 @1400** | **1000** | **1400** | 2.1 |
| compact-pi-mean | 244.2 ± 2.3 | 0.951 | 244.2 @2100 | 2100 (tie) | — | 2.0 |
| compact-wm-actobs-sg | 169.4 ± 3.7 | 0.780 | 172.1 @2000 | — | — | 2.1 |
| compact-baseline-exact | 160.1 ± 4.3 | 0.782 | 167.0 @1900 | — | — | 2.1 |
| compact-pi-obs | 159.2 ± 4.2 | 0.726 | 175.0 @1600 | — | — | 2.8 |
| compact-vocab | 146.6 ± 4.3 | 0.717 | 166.4 @1900 | — | — | 2.0 |
| compact-pi-current | 141.3 ± 4.0 | 0.680 | 187.1 @1600 | — | — | 1.7 |
| compact-pi-seq | 72.7 ± 4.3 | 0.577 | 72.7 @2100 | — | — | 3.4 |
| compact-pi-last | 2.0 ± 2.3 | 0.595 | 3.7 @1900 | — | — | 1.8 |
| grid-base | −20.2 ± 0.5 (@1000, stopped) | 0.211 | −19.8 @800 | — | — | 17.6 |
| grid-vocab | −19.8 ± 0.7 (@900, stopped) | 0.439 | −19.8 @900 | — | — | 19.6 |

Real-environment return by iteration (every snapshot is in `abl2100_evals.json`):

| Run | 300 | 600 | 900 | 1200 | 1500 | 1800 | 2100 |
|---|---|---|---|---|---|---|---|
| compact-base | 13 | 116 | 158 | 138 | 206 | 219 | 244 |
| compact-wm-actobs | 89 | 175 | 233 | 232 | 249 | 218 | 243 |
| compact-pi-mean | 36 | 101 | 165 | 176 | 208 | 228 | 244 |
| compact-wm-actobs-sg | 21 | 95 | 107 | 132 | 142 | 151 | 169 |
| compact-baseline-exact | 9 | 74 | 103 | 99 | 135 | 150 | 160 |
| compact-pi-obs | 66 | 91 | 99 | 136 | 152 | 163 | 159 |
| compact-vocab | −4 | 78 | 90 | 130 | 156 | 160 | 147 |
| compact-pi-current | −9 | 89 | 121 | 123 | 152 | 147 | 141 |
| compact-pi-seq | −16 | 8 | 28 | 28 | 43 | 67 | 73 |
| compact-pi-last | −23 | −19 | −16 | −8 | −7 | −6 | 2 |
| grid-base | −27 | −20 | −22 | — | — | — | — |
| grid-vocab | −25 | −23 | −20 | — | — | — | — |

## Findings by ablation axis

### Latent passed to the heads

**World-model reward/termination heads (`wm-actobs`, `wm-actobs-sg`).** Adding the observation-token mean to the reward/termination head input is the one change that clearly helps. It speeds learning a lot (192 vs 124 at iteration 700; 244 at iteration 1000) and peaks higher (253). In training, reward loss falls faster (0.0026 vs 0.0072 averaged over iterations 300–400) and imagined reward per step rises faster (2.9 vs 1.8 at iterations 700–800).

The stop-gradient control identifies the mechanism. When the heads get the same input but the observation mean is detached, the run lands in the typical range for this config (169). So the gain comes from reward and termination losses training the observation-token latents that the policy reads. In the baseline those losses enter the backbone only at the action token, a position the policy never reads directly. The heads having more information explains little.

This is not a better reward model in general. On exploration data, `wm-actobs`'s reward accuracy trails the others mid-training (0.82–0.84 at iterations 1000–1500) and only catches up by 2100.

**Policy heads (`pi-last`, `pi-mean`).** The observation-token mean carries the useful signal: `pi-mean` ties the baseline (244.2). The last hidden state alone fails almost completely (2.0). `pi-last` also drags its world model down (token accuracy 0.977, reward accuracy 0.715, both the lowest of any compact run), because a policy that keeps crashing fills replay with short, repetitive episodes.

### Vocabulary

Removing the unused grid-cell tokens in compact mode (`compact-vocab`) makes no measurable difference. The world models learn at the same rate (token loss 0.0045 vs 0.0044 at iteration 800), and both converge to the same accuracy. Unused tokens can't affect the input, and cross-entropy pushes their logits down within the first few dozen updates. Sampling is already masked to valid tokens. The return gap (147 vs 244) is within seed spread: `compact-baseline-exact` shares `compact-base`'s vocabulary and reaches only 160. Changing the vocabulary size also changes every random draw, so this comparison is effectively two seeds.

In grid mode, the mode-specific vocabulary didn't help either. Both grid runs failed the same way.

### Mode (grid vs compact)

Grid mode doesn't learn a policy within this budget. By iteration 1000 the policy drives off-road almost immediately (mean episode length 3.75 steps, return ≈ −20, the off-road penalty). The world model is the bottleneck:

- Its 99% token accuracy mostly reflects the static road layout.
- Reward accuracy is 0.2–0.4 and termination accuracy 0.84–0.91 (compact: 0.89 and 1.00 by iteration 1000).
- In imagination it predicts about −10 reward per step, so almost every action looks terminal and PPO has nothing to learn from.

Predicting an outcome from the grid requires the action token to locate the player among 192 cells and check the neighbouring cell. The compact encoding gives row and column directly. Grid mode is also about 10× slower per iteration (17–20 s vs 1.6–2.1 s). The two grid runs were stopped at iterations 1000 and 900, with their snapshots kept.

### Separating the policy from the world model

Both separate actor-critics (their own gpt-mini transformer on raw tokens, trained only by PPO) do worse than reading world-model latents:
- `pi-seq`, with the same 64-step context the world model sees, reaches only 73 and was still slowly improving.
- `pi-obs` (latest observation only) reaches 159, peaking at 175.

A transformer that has to learn its representation from the PPO signal alone learns far more slowly. The world model's latents are shaped by dense next-token, reward, and termination losses on every transition. Separation also doesn't make the world model more accurate: `pi-seq` and `pi-obs` end with the same world-model accuracy as `compact-base`.

### Memory

History is theoretically needed. The compact observation holds task index, position, light, and task text, but not progress within the current task. For "turn left in N units", the policy has to count steps since the task began.

Two memoryless variants were run:
- `pi-obs`: a separate network, so memory is confounded with representation.
- `pi-current`: the same shared world-model latents, encoded from the current observation only.

Their results are consistent with memory helping, but they don't prove it:

| Policy input has history? | Runs (final return) |
|---|---|
| yes | 244 (base), 244 (pi-mean), 243 (wm-actobs), 169 (wm-actobs-sg), 160 (baseline-exact), 147 (vocab) |
| no | 159 (pi-obs, best 175), 141 (pi-current, best 187) |

All three top runs have memory, but several memory runs land in the same range as the memoryless ones.

## World-model accuracy (final snapshot)

Scored on 256 real transitions collected by the road-following random policy. Token accuracy is greedy next-token accuracy on predicted observation tokens. Reward accuracy means the rounded predicted reward equals the true reward.

| Run | Iter | Token acc | Reward acc | Reward MSE | Termination acc |
|---|---|---|---|---|---|
| fullep-v1-mini (old reference) | 2100 | 0.990 | 0.895 | 5.24 | 1.000 |
| compact-base | 2100 | 0.994 | 0.910 | 4.10 | 1.000 |
| compact-wm-actobs | 2100 | 0.996 | 0.910 | 1.92 | 1.000 |
| compact-pi-mean | 2100 | 0.995 | 0.914 | 3.18 | 1.000 |
| compact-wm-actobs-sg | 2100 | 0.993 | 0.840 | 4.07 | 1.000 |
| compact-baseline-exact | 2100 | 0.995 | 0.926 | 0.95 | 1.000 |
| compact-pi-obs | 2100 | 0.993 | 0.914 | 4.01 | 1.000 |
| compact-vocab | 2100 | 0.994 | 0.926 | 4.35 | 1.000 |
| compact-pi-current | 2100 | 0.996 | 0.922 | 3.88 | 1.000 |
| compact-pi-seq | 2100 | 0.995 | 0.930 | 7.59 | 1.000 |
| compact-pi-last | 2100 | 0.977 | 0.715 | 24.45 | 0.988 |
| grid-base | 1000 | 0.988 | 0.387 | 140.15 | 0.840 |
| grid-vocab | 900 | 0.964 | 0.207 | 163.62 | 0.914 |

Healthy compact world models converge to about the same accuracy regardless of policy score, so the world model doesn't explain the return gaps between them. Reward MSE is noisy at 256 transitions: a single missed −20 penalty moves it a lot.

## Caveats

- **One seed per arm.** Runs with the same config spread widely: `compact-base` (autocast on) reaches 244 and `compact-baseline-exact` (autocast off) reaches 160. Changing the vocabulary or any layer shape changes every random draw, and GPU kernels are not deterministic. Falling short of `compact-base`, which sits at the top of that spread, is weak evidence. The `wm-actobs` early lead and the `wm-actobs-sg` collapse are the strongest signals in this sweep.
- **Separate policies** use their own transformer at world-model size with no pretraining. A different size, learning rate, or an auxiliary loss could change the picture.
- **Grid mode:** the original grid runs were stopped early (iterations 1000 and 900). The `grid2` run reached 2100 but is a single seed and had not plateaued.

## Suggested next steps

(Written after the 3-seed runs; step 1 of the original list, the seed reruns, is done.)

1. **Done, then superseded: `compact-vocab-wm-actobs` became the default,** and after the second round `micro-horizon8` (the same with `model_type="gpt-micro"` and `imagination_horizon=8`) replaced it. Bare `DynaConfig()` equals the `micro-horizon8` config, `train.py` and `modal_train.py` default to gpt-micro, and all six sweep specs pin their own settings (52 recorded runs checked). Originally: Across 3 seeds `action+obs` is +26 to +31 at the end, with either vocabulary, and about twice as fast to 200 return; the stop-gradient control confirms its mechanism. Bare `DynaConfig()`, `train.py` and `modal_train.py` now give compact observations, the mode-specific vocabulary, `outcome_features="action+obs"`, gpt-mini, 64 context steps, world batch 64, 64 imagined rollouts and bf16 autocast. The sweep specs pin the old values (`vocab: shared`, `outcome_features: action`), so they still reproduce the recorded runs.
2. **Vocabulary.** Either vocabulary is fine; they're computationally equivalent. Keeping the mode-specific one is tidier.
3. **Grid mode.** The `grid2` run shows that grid mode can learn with the `micro-horizon8` settings, but slowly: 86.5 at 2100 and still rising. Next: resume it past 2100 to find its plateau, run a second seed, and test which change made the difference (`action+obs`, gpt-micro or horizon 8). Helping the world model locate the player (an auxiliary position target, or reading the head at the player's cell) is still the most likely way to speed it up.

## Efficiency changes made for this sweep

Measured on Modal at this config:

| Change | Effect |
|---|---|
| TF32 matmuls + bf16 autocast for world-model training | world-model update phase 1.57 → 0.76 s per iteration (compact, A100) |
| Chunked re-encode of long contexts (`dyna.PREFILL_CHUNK`) | grid mode went from out-of-memory (a 37 GiB attention mask on an 80 GB H100) to 44 GiB peak; identical outputs |
| Plain softmax attention for few-query cached calls (`SelfAttention.SMALL_SCORES`) | grid imagination 69.5 → 37.3 s per iteration; replaces a fused kernel that tiles queries in blocks of 64 |
| bf16 KV cache and world-model inference when `autocast` is on | grid imagination 37.3 → 18.5 s per iteration; checked against fp32 (cosine ≥ 0.999997, identical greedy tokens) |
| Real-environment evaluation moved to separate containers every 100 iterations | removes about 15% of training wall time spent on in-loop evals |
| Exact resume (`train_state.pkl`: replay, env state, RNG, metrics) with automatic handoff before Modal's 24 h limit | long runs continue across calls without the replay-refill dip; a unit test checks a stopped and resumed run matches an uninterrupted one |
| One container per call + `volume.reload()` | fixes two bugs found during smoke tests: a reused container kept the previous run's obs mode (dyna reads `KOTA_OBS` at import) and a stale volume view |

The reference run logged about 8 s per iteration at this stage. These compact runs averaged 1.6–2.1 s per iteration (2.8–3.4 s for separate policies). Grid mode averaged 17–20 s per iteration, down from 86 s when first profiled.

## Reproduce

```
modal deploy modal_train.py
python sweeps/launch.py sweeps/ablations.json [run-name ...]   # spawns train_remote per run on the deployed app
```

Checkpoints, snapshots, `metrics.json`, and `evals/iter_NNNN.json` for each run are under `abl2100/<run>/` on volume `kota-wm-checkpoints`. The baseline checkpoints' evals are under `baseline/fullep-v1-mini/`. Raw evaluation results for every snapshot are in `sweeps/results/abl2100_evals.json`.
