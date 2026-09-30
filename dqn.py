"""Off-policy model-free baseline: Double DQN with replay in the real environment.

Same window encoders (model_free.WindowActorCritic's "gru" / "mlp"), exploration
(Dyna's flow-aware exploration_action, with Dyna's warmup and epsilon) and
evaluator as model_free.py. Unlike PPO, every real transition is kept in a
replay buffer and reused; windows are rebuilt from per-step tokens when sampled.

    python dqn.py --arch gru --total-steps 3000000 --out-dir runs/dqn-gru
"""
import argparse
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from dyna import ACTION_NAMES, OBS_MODE, ORACLE, STEP_LEN, VOCAB_MODE, set_seed
from model_free import (
    PAD,
    ModelFreeConfig,
    VectorCity,
    WindowActorCritic,
    eval_marks,
    evaluate,
    pick_device,
)


@dataclass
class DQNConfig:
    arch: str = "gru"
    context_steps: int = 64
    embed_dim: int = 32
    rnn_dim: int = 256
    hidden: int = 1024
    num_envs: int = 16
    # Gradient updates after each step of all num_envs environments.
    updates_per_step: int = 4
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    n_step: int = 3
    lr: float = 1e-4
    gamma: float = 0.995
    target_update: int = 2000
    max_grad_norm: float = 10.0
    reward_scale: float = 20.0
    max_episode_steps: int = 256
    # Dyna's collector: warmup_steps of pure exploration, then epsilon.
    warmup_steps: int = 1024
    epsilon: float = 0.3
    random_action_fraction: float = 0.1

    def __post_init__(self):
        if self.arch not in ("gru", "mlp"):
            raise ValueError("arch must be 'gru' or 'mlp'")
        if not 0 <= self.epsilon <= 1 or not 0 <= self.random_action_fraction <= 1:
            raise ValueError("epsilon and random_action_fraction must lie in [0, 1]")
        if min(self.num_envs, self.batch_size, self.n_step, self.target_update, self.context_steps) < 1:
            raise ValueError("Sizes must be positive")
        if self.replay_capacity // self.num_envs <= self.context_steps + self.n_step + 1:
            raise ValueError("replay_capacity too small for the window and n-step lookahead")


class WindowQNetwork(WindowActorCritic):
    """Dueling Q-network on WindowActorCritic's encoder: Q = V + A - mean(A).

    forward returns (Categorical(logits=Q), Q) so model_free.evaluate's greedy
    argmax picks argmax Q."""

    def q_values(self, windows):
        h = self.features(windows)
        advantage = self.pi_head(h)
        return self.v_head(h) + advantage - advantage.mean(dim=-1, keepdim=True)

    def forward(self, windows):
        q = self.q_values(windows)
        return Categorical(logits=q), q


class StepReplay:
    """Per-env ring buffers of steps, each with its own head. Slot t holds the step
    tokens [prev action, obs] of a state and, once acted on, its action, reward and
    termination. A time-limit cutoff writes the final state as a slot that is never
    acted on (`acted` False), so n-step targets bootstrap from it. Each env's
    episodes are contiguous in its buffer."""

    def __init__(self, num_envs, capacity, device):
        self.E, self.C, self.device = num_envs, capacity // num_envs, device
        E, C = self.E, self.C
        self.tokens = torch.full((E, C, STEP_LEN), PAD, dtype=torch.uint8, device=device)
        self.action = torch.zeros(E, C, dtype=torch.long, device=device)
        self.reward = torch.zeros(E, C, device=device)
        self.done = torch.zeros(E, C, dtype=torch.bool, device=device)
        self.acted = torch.zeros(E, C, dtype=torch.bool, device=device)
        self.episode = torch.full((E, C), -1, dtype=torch.long, device=device)
        # Absolute index of each env's current (not yet acted) slot.
        self.head = torch.zeros(E, dtype=torch.long, device=device)
        self.next_episode = 0
        self.env_episode = torch.zeros(E, dtype=torch.long, device=device)

    def _write(self, env_ids, tokens):
        slot = self.head[env_ids] % self.C
        self.tokens[env_ids, slot] = tokens.to(torch.uint8)
        self.episode[env_ids, slot] = self.env_episode[env_ids]
        self.acted[env_ids, slot] = False

    def start(self, env_ids, tokens):
        """New episodes in env_ids, their first states written at the heads."""
        n = len(env_ids)
        self.env_episode[env_ids] = torch.arange(self.next_episode, self.next_episode + n, device=self.device)
        self.next_episode += n
        self._write(env_ids, tokens)

    def continue_(self, env_ids, tokens):
        """The next state of an ongoing episode."""
        self._write(env_ids, tokens)

    def act(self, actions, rewards, dones):
        """Record every env's action at its head slot and advance."""
        envs = torch.arange(self.E, device=self.device)
        slot = self.head % self.C
        self.action[envs, slot], self.reward[envs, slot], self.done[envs, slot] = actions, rewards, dones
        self.acted[envs, slot] = True
        self.head += 1

    def cutoff(self, env_ids, tokens):
        """Time-limit final states for env_ids: unacted slots, then advance those envs."""
        self._write(env_ids, tokens)
        self.head[env_ids] += 1

    def windows(self, env, t, K):
        """(B, K, STEP_LEN) windows ending at absolute slots t of envs `env`; steps of
        other episodes (or overwritten / never written) become <PAD>."""
        idx = t[:, None] - torch.arange(K - 1, -1, -1, device=self.device)  # (B, K), absolute
        slots = idx % self.C
        tokens = self.tokens[env[:, None], slots].long()
        same = self.episode[env[:, None], slots] == self.episode[env, t % self.C][:, None]
        same &= (idx >= 0) & (idx > self.head[env][:, None] - self.C)
        return torch.where(same[..., None], tokens, torch.full_like(tokens, PAD))

    def sample(self, batch, K, n, gamma):
        """Windows, actions, n-step discounted rewards, bootstrap windows and their discounts."""
        env = torch.randint(self.E, (2 * batch,), device=self.device)
        head = self.head[env]
        lo = (head - self.C + K + n + 1).clamp(min=0)
        hi = head - n  # exclusive: slots t..t+n are all written, none is the unacted head
        valid = hi > lo
        t = lo + (torch.rand(len(env), device=self.device) * (hi - lo).clamp(min=1)).long()
        ok = valid & self.acted[env, t % self.C]
        env, t = env[ok][:batch], t[ok][:batch]
        if len(t) == 0:
            return None
        return self.targets(env, t, K, n, gamma)

    def targets(self, env, t, K, n, gamma):
        """For acted slots t of envs `env`: windows, actions, n-step returns, bootstrap windows, discounts."""
        B = len(t)
        ret = torch.zeros(B, device=self.device)
        alive = torch.ones(B, dtype=torch.bool, device=self.device)
        boot_t = t + n
        boot_discount = torch.full((B,), gamma**n, device=self.device)
        for k in range(n):
            s = (t + k) % self.C
            cut = alive & ~self.acted[env, s]  # a time-limit final state: bootstrap from it
            boot_t = torch.where(cut, t + k, boot_t)
            boot_discount = torch.where(cut, torch.full_like(boot_discount, gamma**k), boot_discount)
            alive &= ~cut
            ret += alive * gamma**k * self.reward[env, s]
            ended = alive & self.done[env, s]
            boot_discount = torch.where(ended, torch.zeros_like(boot_discount), boot_discount)
            alive &= ~ended
        return (
            self.windows(env, t, K),
            self.action[env, t % self.C],
            ret,
            self.windows(env, boot_t, K),
            boot_discount,
        )


def train(config, total_steps, out_dir, seed=3, eval_episodes=200, eval_every=250_000, device=None, on_eval=None):
    set_seed(seed)
    device = torch.device(device) if device is not None else pick_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    c = config
    encoder = ModelFreeConfig(arch=c.arch, context_steps=c.context_steps, embed_dim=c.embed_dim,
                              rnn_dim=c.rnn_dim, hidden=c.hidden, max_episode_steps=c.max_episode_steps)
    q = WindowQNetwork(encoder).to(device)
    target = WindowQNetwork(encoder).to(device)
    target.load_state_dict(q.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(q.parameters(), lr=c.lr, eps=1e-5)
    N, K = c.num_envs, c.context_steps
    envs = VectorCity(N, K, c.max_episode_steps)
    replay = StepReplay(N, c.replay_capacity, device)
    replay.start(torch.arange(N, device=device), torch.from_numpy(envs.windows[:, -1]).to(device))
    marks = eval_marks(total_steps, 16_384, eval_every)
    next_mark = 0
    results = dict(algo="dqn", config=asdict(c), obs_mode=OBS_MODE, vocab=VOCAB_MODE, oracle=ORACLE, seed=seed,
                   total_steps=total_steps, parameters=sum(p.numel() for p in q.parameters()),
                   updates=[], evals=[], best_return=float("-inf"), best_steps=None)
    print(f"Device {device}; {results['parameters']:,} parameters; config {c}", flush=True)
    steps = gradient_steps = 0
    recent, losses = [], []
    started = time.time()

    def run_eval():
        result = evaluate(q, encoder, eval_episodes)
        result.update(env_steps=steps, elapsed=time.time() - started, gradient_steps=gradient_steps)
        results["evals"].append(result)
        if result["mean_return"] > results["best_return"]:
            results["best_return"], results["best_steps"] = result["mean_return"], steps
            torch.save(dict(config=asdict(c), q=q.state_dict(), env_steps=steps), out_dir / "q_best.pt")
        print("Eval:", json.dumps(result), flush=True)
        (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))
        if on_eval is not None:
            on_eval()

    while steps < total_steps:
        explore = torch.rand(N) < (1.0 if steps < c.warmup_steps else c.epsilon)
        probs = torch.from_numpy(envs.exploration_probs(c.random_action_fraction))
        actions = torch.multinomial(probs, 1).squeeze(1)
        if not explore.all():
            with torch.no_grad():
                greedy = q.q_values(torch.from_numpy(envs.windows).to(device)).argmax(-1).cpu()
            actions = torch.where(explore, actions, greedy)
        rewards, terminated, truncated, final, finished, _, _ = envs.step(actions.tolist())
        replay.act(actions.to(device), torch.from_numpy(rewards).to(device) / c.reward_scale,
                   torch.from_numpy(terminated).to(device))
        if final:
            ids = sorted(final)
            replay.cutoff(torch.tensor(ids, device=device),
                          torch.from_numpy(np.stack([final[i][-1] for i in ids])).to(device))
        ended = np.flatnonzero(terminated | truncated)
        going = np.flatnonzero(~(terminated | truncated))
        latest = torch.from_numpy(envs.windows[:, -1]).to(device)
        if len(going):
            replay.continue_(torch.from_numpy(going).to(device), latest[going])
        if len(ended):
            replay.start(torch.from_numpy(ended).to(device), latest[ended])
        recent.extend(finished)
        steps += N

        if steps >= c.warmup_steps:
            for _ in range(c.updates_per_step):
                batch = replay.sample(c.batch_size, K, c.n_step, c.gamma)
                if batch is None:
                    break
                windows, action, ret, next_windows, discount = batch
                with torch.no_grad():
                    best = q.q_values(next_windows).argmax(-1, keepdim=True)
                    target_q = ret + discount * target.q_values(next_windows).gather(1, best).squeeze(1)
                predicted = q.q_values(windows).gather(1, action[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(predicted, target_q)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), c.max_grad_norm)
                optimizer.step()
                gradient_steps += 1
                losses.append(loss.item())
                if gradient_steps % c.target_update == 0:
                    target.load_state_dict(q.state_dict())

        if steps % (N * 1000) == 0:
            recent = recent[-200:]
            update = dict(env_steps=steps, elapsed=time.time() - started, gradient_steps=gradient_steps,
                          train_return=float(np.mean([r for r, _ in recent])) if recent else None,
                          train_length=float(np.mean([n for _, n in recent])) if recent else None,
                          loss=float(np.mean(losses)) if losses else None)
            losses = []
            results["updates"].append(update)
            print(update, flush=True)
        while next_mark < len(marks) and steps >= marks[next_mark]:
            next_mark += 1
            if next_mark == len(marks) or steps < marks[next_mark]:
                run_eval()
    if not results["evals"] or results["evals"][-1]["env_steps"] != steps:
        run_eval()
    torch.save(dict(config=asdict(c), q=q.state_dict(), env_steps=steps), out_dir / "q.pt")
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--total-steps", type=int, default=3_000_000)
    parser.add_argument("--out-dir", default="runs/dqn")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=250_000)
    parser.add_argument("--device", default=None)
    for field in fields(DQNConfig):
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=field.type, default=field.default)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = DQNConfig(**{f.name: getattr(args, f.name) for f in fields(DQNConfig)})
    train(config, args.total_steps, args.out_dir, args.seed, args.eval_episodes, args.eval_every, args.device)


if __name__ == "__main__":
    main()
