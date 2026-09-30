"""Model-free PPO in the real environment (no world model, no imagination).

The policy sees the last `context_steps` steps, the same window the world model
reads (DynaConfig.context_steps). Each step is [previous action, observation]
tokens. Every sample the policy trains on comes from the real environment, so
`total_steps` is directly comparable with Dyna's real_steps.

    python model_free.py --arch gru --total-steps 20000000 --out-dir runs/mf-gru
"""
import argparse
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from dyna import (
    ACTION_NAMES,
    ACTION_TOKEN_IDS,
    OBS_MODE,
    ORACLE,
    STEP_LEN,
    VOCAB,
    VOCAB_MODE,
    VOCAB_SIZE,
    observe,
    flow_actions,
    random_city,
    set_seed,
)

PAD = VOCAB["<PAD>"]
ARCHS = ("gru", "mlp")


@dataclass
class ModelFreeConfig:
    arch: str = "gru"
    # 64 = DynaConfig.context_steps; 1 gives a memoryless policy (current step only).
    context_steps: int = 64
    embed_dim: int = 32
    rnn_dim: int = 256
    hidden: int = 1024
    num_envs: int = 64
    rollout_steps: int = 128
    epochs: int = 4
    minibatch_size: int = 1024
    lr: float = 1e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    value_clip: float = 0.2
    entropy_coef: float = 0.02
    max_grad_norm: float = 0.5
    reward_scale: float = 20.0
    max_episode_steps: int = 256
    # Acting (and PPO's likelihoods) use the mixture (1 - explore_eps) * pi +
    # explore_eps * q, where q is dyna.exploration_action's distribution: uniform
    # over traffic-flow moves and no-op, uniform over all actions with
    # probability random_action_fraction. Dyna's collector uses 0.3 / 0.1.
    # Evaluation is greedy on pi alone. 0 is plain PPO.
    explore_eps: float = 0.0
    random_action_fraction: float = 0.1
    # The first warmup_steps real steps act from the exploration distribution
    # alone (explore_eps = 1), like Dyna's collector warmup (DynaConfig.warmup_steps).
    warmup_steps: int = 0

    def __post_init__(self):
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}")
        if not 0 <= self.explore_eps <= 1 or not 0 <= self.random_action_fraction <= 1:
            raise ValueError("explore_eps and random_action_fraction must lie in [0, 1]")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if min(self.context_steps, self.num_envs, self.rollout_steps, self.epochs, self.minibatch_size) < 1:
            raise ValueError("Sizes must be positive")


class WindowActorCritic(nn.Module):
    """Actor-critic over a (B, K, STEP_LEN) window of tokens, whole steps left-padded with <PAD>.

    "gru" embeds each step and runs a GRU across the window, reading its last state;
    "mlp" flattens every token embedding of the window into one input vector.
    """

    def __init__(self, config, n_actions=len(ACTION_NAMES)):
        super().__init__()
        self.arch = config.arch
        self.embed = nn.Embedding(VOCAB_SIZE, config.embed_dim)
        step_dim = STEP_LEN * config.embed_dim
        if config.arch == "gru":
            self.step_proj = nn.Sequential(nn.Linear(step_dim, config.rnn_dim), nn.GELU())
            self.rnn = nn.GRU(config.rnn_dim, config.rnn_dim, batch_first=True)
            features = config.rnn_dim
        else:
            features = config.context_steps * step_dim
        self.body = nn.Sequential(
            nn.Linear(features, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, config.hidden),
            nn.GELU(),
        )
        self.pi_head = nn.Linear(config.hidden, n_actions)
        self.v_head = nn.Linear(config.hidden, 1)
        nn.init.orthogonal_(self.pi_head.weight, gain=0.01)
        nn.init.constant_(self.pi_head.bias, 0.0)

    def features(self, windows):
        x = self.embed(windows).flatten(2)  # (B, K, STEP_LEN * embed_dim)
        if self.arch == "gru":
            h, _ = self.rnn(self.step_proj(x))
            x = h[:, -1]
        else:
            x = x.flatten(1)
        return self.body(x)

    def forward(self, windows):
        h = self.features(windows)
        return Categorical(logits=self.pi_head(h)), self.v_head(h).squeeze(-1)


def mixture(dist, explore_probs, eps):
    """The acting distribution: pi mixed with the exploration distribution.

    `eps` is a number or a (B,) tensor of per-row mixing weights."""
    if torch.is_tensor(eps):
        eps = eps[:, None]
    elif eps == 0:
        return dist
    return Categorical(probs=(1 - eps) * dist.probs + eps * explore_probs)


class VectorCity:
    """`n` independent City episodes, each exposed as its last `context_steps` steps."""

    def __init__(self, n, context_steps, max_episode_steps):
        self.n, self.context_steps, self.max_episode_steps = n, context_steps, max_episode_steps
        self.windows = np.full((n, context_steps, STEP_LEN), PAD, dtype=np.int64)
        self.cities = [None] * n
        self.steps = [0] * n
        self.returns = [0.0] * n
        for i in range(n):
            self._reset(i)

    def _reset(self, i):
        city = random_city()
        self.cities[i] = city
        self.steps[i], self.returns[i] = 0, 0.0
        self.windows[i] = PAD
        self.windows[i, -1] = [PAD] + observe(city)

    def exploration_probs(self, random_fraction):
        """(n, actions) distribution of dyna.exploration_action in each env (uniform if done)."""
        n_actions = len(ACTION_NAMES)
        probs = np.full((self.n, n_actions), 1.0 / n_actions, dtype=np.float32)
        for i, city in enumerate(self.cities):
            if city is None:
                continue
            flow = np.zeros(n_actions, dtype=np.float32)
            flow[flow_actions(city)] = 1.0
            probs[i] = random_fraction / n_actions + (1 - random_fraction) * flow / flow.sum()
        return probs

    def step(self, actions, auto_reset=True):
        """Returns rewards, terminated, truncated, the windows after truncated steps
        (for bootstrapping, keyed by env index), finished episodes and task counts."""
        rewards = np.zeros(self.n, dtype=np.float32)
        terminated = np.zeros(self.n, dtype=bool)
        truncated = np.zeros(self.n, dtype=bool)
        final_windows, finished = {}, []
        completed = deviated = 0
        for i, a in enumerate(actions):
            city = self.cities[i]
            if city is None:
                continue
            expected = city._task_directions[city._dir_idx]
            reward, term = city.step(ACTION_NAMES[a])
            if city._need_new_task:
                completed += ACTION_NAMES[a] == expected
                deviated += ACTION_NAMES[a] != expected
            rewards[i] = reward
            self.returns[i] += float(reward)
            self.steps[i] += 1
            trunc = not term and self.steps[i] >= self.max_episode_steps
            terminated[i], truncated[i] = term, trunc
            if not term:
                city.env_step()
                window = np.roll(self.windows[i], -1, axis=0)
                window[-1] = [ACTION_TOKEN_IDS[a]] + observe(city)
                self.windows[i] = window
            if term or trunc:
                finished.append((self.returns[i], self.steps[i]))
                if trunc:
                    final_windows[i] = self.windows[i].copy()
                if auto_reset:
                    self._reset(i)
                else:
                    self.cities[i] = None
        return rewards, terminated, truncated, final_windows, finished, completed, deviated



def _rng_state():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    )


def _set_rng_state(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3] is not None:
        torch.cuda.set_rng_state_all(state[3])


@torch.no_grad()
def evaluate(policy, config, episodes=200, seed=0):
    """Greedy episodes, like evaluate.run_episodes, run side by side. Training RNG is untouched."""
    saved = _rng_state()
    set_seed(seed)
    device = next(policy.parameters()).device
    was_training = policy.training
    policy.eval()
    envs = VectorCity(episodes, config.context_steps, config.max_episode_steps)
    returns, lengths, completed, deviated = [], [], 0, 0
    try:
        while any(c is not None for c in envs.cities):
            dist, _ = policy(torch.from_numpy(envs.windows).to(device))
            actions = dist.probs.argmax(-1).tolist()
            *_, finished, done_ok, done_bad = envs.step(actions, auto_reset=False)
            for r, n in finished:
                returns.append(r)
                lengths.append(n)
            completed += done_ok
            deviated += done_bad
    finally:
        policy.train(was_training)
        _set_rng_state(saved)
    returns_t = torch.tensor(returns)
    return dict(
        episodes=len(returns),
        mean_return=returns_t.mean().item(),
        std_return=returns_t.std().item(),
        sem_return=(returns_t.std() / len(returns) ** 0.5).item(),
        mean_length=sum(lengths) / len(lengths),
        tasks_completed=completed,
        tasks_deviated=deviated,
        completion_rate=completed / max(completed + deviated, 1),
    )


def eval_marks(total_steps, batch, every):
    """Doubling from one batch up to `every`, then every `every` steps."""
    marks, m = [], batch
    while m < every:
        marks.append(m)
        m *= 2
    marks.extend(range(every, total_steps + every, every))
    return marks


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(
    config,
    total_steps,
    out_dir,
    seed=3,
    eval_episodes=200,
    eval_every=500_000,
    device=None,
    on_eval=None,
):
    set_seed(seed)
    device = torch.device(device) if device is not None else pick_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = WindowActorCritic(config).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.lr, eps=1e-5)
    c = config
    N, T = c.num_envs, c.rollout_steps
    envs = VectorCity(N, c.context_steps, c.max_episode_steps)
    batch = N * T
    marks = eval_marks(total_steps, batch, eval_every)
    next_mark = 0
    results = dict(
        config=asdict(c),
        obs_mode=OBS_MODE,
        vocab=VOCAB_MODE,
        oracle=ORACLE,
        seed=seed,
        total_steps=total_steps,
        parameters=sum(p.numel() for p in policy.parameters()),
        updates=[],
        evals=[],
        best_return=float("-inf"),
        best_steps=None,
    )
    print(f"Device {device}; {results['parameters']:,} parameters; config {c}")
    steps = 0
    recent = []  # (return, length) of recent training episodes
    started = time.time()

    def run_eval():
        result = evaluate(policy, c, eval_episodes)
        result.update(env_steps=steps, elapsed=time.time() - started)
        results["evals"].append(result)
        if result["mean_return"] > results["best_return"]:
            results["best_return"], results["best_steps"] = result["mean_return"], steps
            torch.save(dict(config=asdict(c), policy=policy.state_dict(), env_steps=steps),
                       out_dir / "policy_best.pt")
        print("Eval:", json.dumps(result), flush=True)
        (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))
        if on_eval is not None:
            on_eval()

    shape = (T, N, c.context_steps, STEP_LEN)
    obs_buf = torch.zeros(shape, dtype=torch.uint8)
    act_buf = torch.zeros(T, N, dtype=torch.long)
    explore_buf = torch.zeros(T, N, len(ACTION_NAMES))
    eps_buf = torch.zeros(T, N)
    logp_buf = torch.zeros(T, N)
    val_buf = torch.zeros(T, N)
    rew_buf = torch.zeros(T, N)
    done_buf = torch.zeros(T, N)
    while steps < total_steps:
        policy.eval()
        with torch.no_grad():
            for t in range(T):
                windows = torch.from_numpy(envs.windows)
                obs_buf[t] = windows.to(torch.uint8)
                dist, value = policy(windows.to(device))
                explore = torch.from_numpy(envs.exploration_probs(c.random_action_fraction))
                explore_buf[t] = explore
                eps_buf[t] = 1.0 if steps + t * N < c.warmup_steps else c.explore_eps
                dist = mixture(dist, explore.to(device), eps_buf[t].to(device))
                action = dist.sample()
                act_buf[t] = action.cpu()
                logp_buf[t] = dist.log_prob(action).cpu()
                val_buf[t] = value.float().cpu()
                rewards, terminated, truncated, final, finished, _, _ = envs.step(action.tolist())
                rewards = torch.from_numpy(rewards) / c.reward_scale
                if final:
                    # Time limit, not a real ending: bootstrap from the state it cut off.
                    idx = list(final)
                    _, v = policy(torch.from_numpy(np.stack([final[i] for i in idx])).to(device))
                    rewards[idx] += c.gamma * v.float().cpu()
                rew_buf[t] = rewards
                done_buf[t] = torch.from_numpy(terminated | truncated).float()
                recent.extend(finished)
            _, last_value = policy(torch.from_numpy(envs.windows).to(device))
            last_value = last_value.float().cpu()
        steps += batch

        advantages = torch.zeros(T, N)
        running = torch.zeros(N)
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else val_buf[t + 1]
            nonterminal = 1.0 - done_buf[t]
            delta = rew_buf[t] + c.gamma * next_value * nonterminal - val_buf[t]
            running = delta + c.gamma * c.gae_lambda * nonterminal * running
            advantages[t] = running
        returns = advantages + val_buf

        b_obs = obs_buf.reshape(batch, c.context_steps, STEP_LEN).to(device).long()
        b_act, b_logp = act_buf.reshape(-1).to(device), logp_buf.reshape(-1).to(device)
        b_val, b_ret = val_buf.reshape(-1).to(device), returns.reshape(-1).to(device)
        b_adv = advantages.reshape(-1).to(device)
        b_explore = explore_buf.reshape(batch, -1).to(device)
        b_eps = eps_buf.reshape(-1).to(device)
        policy.train()
        stats = []
        for _ in range(c.epochs):
            for mb in torch.randperm(batch, device=device).split(c.minibatch_size):
                pi, value = policy(b_obs[mb])
                dist = mixture(pi, b_explore[mb], b_eps[mb])
                adv = b_adv[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                ratio = (dist.log_prob(b_act[mb]) - b_logp[mb]).exp()
                actor = -torch.minimum(ratio * adv, ratio.clamp(1 - c.clip_eps, 1 + c.clip_eps) * adv).mean()
                critic = (value - b_ret[mb]) ** 2
                if c.value_clip:
                    clipped = b_val[mb] + (value - b_val[mb]).clamp(-c.value_clip, c.value_clip)
                    critic = torch.maximum(critic, (clipped - b_ret[mb]) ** 2)
                critic = critic.mean()
                entropy = pi.entropy().mean()
                loss = actor + c.value_coef * critic - c.entropy_coef * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), c.max_grad_norm)
                optimizer.step()
                with torch.no_grad():
                    kl = (b_logp[mb] - dist.log_prob(b_act[mb])).mean()
                stats.append((actor.item(), critic.item(), entropy.item(), kl.item()))
        actor, critic, entropy, kl = np.mean(stats, axis=0)
        recent = recent[-200:]
        update = dict(
            env_steps=steps,
            elapsed=time.time() - started,
            train_return=float(np.mean([r for r, _ in recent])) if recent else None,
            train_length=float(np.mean([n for _, n in recent])) if recent else None,
            actor_loss=float(actor),
            critic_loss=float(critic),
            entropy=float(entropy),
            approx_kl=float(kl),
        )
        results["updates"].append(update)
        if len(results["updates"]) % 10 == 1:
            print(update, flush=True)
        while next_mark < len(marks) and steps >= marks[next_mark]:
            next_mark += 1
            if next_mark == len(marks) or steps < marks[next_mark]:
                run_eval()
    if not results["evals"] or results["evals"][-1]["env_steps"] != steps:
        run_eval()
    torch.save(dict(config=asdict(c), policy=policy.state_dict(), env_steps=steps),
               out_dir / "policy.pt")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--total-steps", type=int, default=20_000_000)
    parser.add_argument("--out-dir", default="runs/model-free")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=500_000)
    parser.add_argument("--device", default=None)
    for field in fields(ModelFreeConfig):
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=field.type, default=field.default)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = ModelFreeConfig(**{f.name: getattr(args, f.name) for f in fields(ModelFreeConfig)})
    train(config, args.total_steps, args.out_dir, args.seed, args.eval_episodes, args.eval_every, args.device)


if __name__ == "__main__":
    main()
