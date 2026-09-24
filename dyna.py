from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from envs.full.env import City
from models.actor_critic import ActorCritic
from models.transformer import TransformerConfig
from models.world_model import WorldModel

OBS_MODE = os.environ.get("KOTA_OBS", "grid")
if OBS_MODE not in ("grid", "compact"):
    raise ValueError(f"KOTA_OBS must be 'grid' or 'compact', not {OBS_MODE!r}")

_STATIC_CITY = City(spawn_point=(0, 0))

# task is reachable if it follows the graph out-edge flow from curr node
REACHABLE_TASKS = {
    node: frozenset(text for text, _ in _STATIC_CITY.task_options(*node))
    for node in _STATIC_CITY.graph.nodes
}


def _ordinal(k):
    return f"{k}{'st' if k == 1 else 'nd' if k == 2 else 'rd' if k == 3 else 'th'}"


# each task is one token
_ISSUABLE = frozenset().union(*REACHABLE_TASKS.values())
TASK_TEXTS = tuple(
    text
    for text in [
        f"Turn {side} in {n} unit{'s' if n > 1 else ''}."
        for side in ("left", "right")
        for n in range(1, 16)
    ]
    + [
        f"Turn {side} on {_ordinal(k)} {road}."
        for side in ("left", "right")
        for road, count in (("Avenue", 4), ("Street", 6))
        for k in range(1, count + 1)
    ]
    if text in _ISSUABLE
)
assert set(TASK_TEXTS) == _ISSUABLE, "City issues invalid task"

TOKENS = [
    # obs
    "0",
    "1",
    "A",
    "S",
    "P",
    # act
    "W_act",
    "A_act",
    "S_act",
    "D_act",
    "NOOP",
    # task index
    *[f"task {k}" for k in range(1, 11)],
    # task
    *TASK_TEXTS,
    # compact obs
    *[f"row {r}" for r in range(16)],
    *[f"col {c}" for c in range(12)],
    "stop A",
    "stop S",
    # pad
    "<PAD>",
]

VOCAB = {token: i for i, token in enumerate(TOKENS)}
VOCAB_SIZE = len(VOCAB)

assert VOCAB_SIZE == 94

GRID_ROWS, GRID_COLS = 16, 12
# task index (1) + either the 16 x 12 grid or row/col/stop + task (1)
OUT_LEN = 1 + (GRID_ROWS * GRID_COLS if OBS_MODE == "grid" else 3) + 1
# The task is not predicted by the world model, so it comes last: a generated
# observation draws a new task from the ones City can issue at the generated
# player position exactly when the generated task index advances, which needs
# both the index and the grid (or row/col) first.
TASK_INDEX_POS, TASK_POS = 0, OUT_LEN - 1

assert City.MAX_TASKS == 10
TASK_INDEX_TOKEN_IDS = tuple(VOCAB[f"task {k}"] for k in range(1, City.MAX_TASKS + 1))
TASK_TOKEN_IDS = tuple(VOCAB[t] for t in TASK_TEXTS)
ROW_TOKEN_IDS = tuple(VOCAB[f"row {r}"] for r in range(GRID_ROWS))
COL_TOKEN_IDS = tuple(VOCAB[f"col {c}"] for c in range(GRID_COLS))
STOP_TOKEN_IDS = (VOCAB["stop A"], VOCAB["stop S"])
CELL_TOKEN_IDS = tuple(VOCAB[t] for t in ("0", "1", "A", "S", "P"))

ACTION_NAMES = ("W", "A", "S", "D", "")
ACTION_TOKEN_IDS = tuple(VOCAB[t] for t in ("W_act", "A_act", "S_act", "D_act", "NOOP"))
ACTION_DELTAS = ((-1, 0), (0, -1), (1, 0), (0, 1), (0, 0))
STEP_LEN = OUT_LEN + 1
CTX_LEN = 24 * STEP_LEN

# (num_layers, num_heads, embed_dim)
MODEL_PRESETS = {
    "gpt-nano": (3, 3, 48),
    "gpt-micro": (4, 4, 128),
    "gpt-mini": (6, 6, 192),
    "gopher-44m": (8, 16, 512),
    "gpt2": (12, 12, 768),
    "gpt2-medium": (24, 16, 1024),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_city():
    while True:
        try:
            city = City(
                spawn_point=(random.randrange(GRID_ROWS), random.randrange(GRID_COLS))
            )
            break
        except AssertionError:  # not a road
            continue
    city.env_step()
    return city


def flow_actions(city):
    here = (city.p_row, city.p_col)
    return [
        i
        for i, (dr, dc) in enumerate(ACTION_DELTAS)
        if (dr, dc) == (0, 0) or city.graph.has_edge(here, (here[0] + dr, here[1] + dc))
    ]


def exploration_action(city, random_fraction):
    if random.random() < random_fraction:
        return random.randrange(len(ACTION_NAMES))
    return random.choice(flow_actions(city))


def tokenize_task(task_text):
    if task_text not in VOCAB or VOCAB[task_text] not in TASK_TOKEN_IDS:
        raise ValueError(f"Cannot tokenize task: {task_text!r}")
    return VOCAB[task_text]


def player_position(tokens):
    if OBS_MODE == "compact":
        return ROW_TOKEN_IDS.index(tokens[1]), COL_TOKEN_IDS.index(tokens[2])
    cells = list(tokens[1 : 1 + GRID_ROWS * GRID_COLS])
    if cells.count(VOCAB["P"]) != 1:
        return None
    return divmod(cells.index(VOCAB["P"]), GRID_COLS)


def sample_task_token(position):
    if position in REACHABLE_TASKS:
        return VOCAB[_STATIC_CITY.generate_task(*position)[0]]
    return random.choice(TASK_TOKEN_IDS)


def observe(city):
    if not 1 <= city._task_idx <= City.MAX_TASKS:
        raise ValueError("Observe after env_step, once the first task exists")
    if OBS_MODE == "compact":
        body = [
            VOCAB[f"row {city.p_row}"],
            VOCAB[f"col {city.p_col}"],
            VOCAB[f"stop {city.stop_road}"],
        ]
    else:
        body = [VOCAB[str(v)] for row in city.grid for v in row]
    return [VOCAB[f"task {city._task_idx}"]] + body + [tokenize_task(city.task)]


class Episode:
    def __init__(self, first_observation):
        self.observations = [tuple(first_observation)]
        self.actions, self.rewards = [], []
        # Whether the last action ended the episode; no time limits
        self.terminated = False

    def __len__(self):
        return len(self.actions)

    def tokens(self, start, end, with_action=False):
        out = list(self.observations[start])
        for t in range(start, end):
            out.append(ACTION_TOKEN_IDS[self.actions[t]])
            out.extend(self.observations[t + 1])
        if with_action:
            out.append(ACTION_TOKEN_IDS[self.actions[end]])
        return out


class Replay:
    def __init__(self, capacity=None):
        self.capacity = capacity
        self.episodes = deque()
        self.size = 0

    def __len__(self):
        return self.size

    def start(self, observation):
        self.episodes.append(Episode(observation))

    def add(self, action_idx, reward, terminated, next_observation):
        episode = self.episodes[-1]
        episode.actions.append(action_idx)
        episode.rewards.append(reward)
        episode.observations.append(tuple(next_observation))
        episode.terminated = terminated
        self.size += 1
        while (
            self.capacity is not None
            and self.size > self.capacity
            and len(self.episodes) > 1
        ):
            self.size -= len(self.episodes.popleft())

    def sample_steps(self, k):
        episodes = [e for e in self.episodes if len(e)]
        if not episodes:
            raise ValueError("Replay holds no transitions")
        chosen = random.choices(episodes, weights=[len(e) for e in episodes], k=k)
        return [(e, random.randrange(len(e))) for e in chosen]


class CachedContext:
    def __init__(self, world_model, rows):
        self.world_model = world_model
        self.device = next(world_model.parameters()).device
        self.reset(rows)

    @staticmethod
    def _rows(rows):
        rows = list(rows)
        if rows and not isinstance(rows[0], (list, tuple)):
            rows = [rows]
        rows = [list(r) for r in rows]
        if not rows or not all(rows):
            raise ValueError("Context rows must be non-empty")
        return rows

    @property
    def batch_size(self):
        return len(self.tokens)

    @property
    def length(self):
        """Padded length: the longest row's."""
        return len(self.tokens[0]) + self.pad[0]

    @torch.no_grad()
    def reset(self, rows):
        self.tokens = self._rows(rows)
        width = max(len(r) for r in self.tokens)
        self.pad = [width - len(r) for r in self.tokens]
        self.pad_tensor = (
            torch.tensor(self.pad, device=self.device) if any(self.pad) else None
        )
        self.keys_values = self.world_model.generate_empty_keys_values(self.batch_size)
        idx = torch.tensor(
            [[VOCAB["<PAD>"]] * p + r for p, r in zip(self.pad, self.tokens)],
            dtype=torch.long,
            device=self.device,
        )
        h = self.world_model(idx, self.keys_values, self.pad_tensor)
        self.hidden = h[:, -1, :]
        self.obs_hidden = h[:, -OUT_LEN:, :]

    @torch.no_grad()
    def append(self, rows):
        if torch.is_tensor(rows):
            rows = rows.tolist()
        rows = list(rows)
        if not rows:
            return
        if not isinstance(rows[0], (list, tuple)):
            rows = [[t] for t in rows] if self.batch_size > 1 else [rows]
        rows = [list(r) for r in rows]
        if len(rows) != self.batch_size or any(len(r) != len(rows[0]) for r in rows):
            raise ValueError("Append one equal-length token list per row")
        if not rows[0]:
            return
        if self.length + len(rows[0]) > self.world_model.max_tokens:
            raise ValueError("Crop the context before appending beyond max_tokens")
        idx = torch.tensor(rows, dtype=torch.long, device=self.device)
        h = self.world_model(idx, self.keys_values, self.pad_tensor)
        for row, new in zip(self.tokens, rows):
            row.extend(new)
        self.hidden = h[:, -1, :]
        self.obs_hidden = torch.cat((self.obs_hidden, h), dim=1)[:, -OUT_LEN:, :]

    def crop(self, length):
        """Keep at most `length` trailing tokens per row (shorter rows lose only padding)."""
        if self.length > length:
            self.reset([row[-length:] for row in self.tokens])

    def allowed_mask(self, token_ids):
        """(B, V) mask allowing the same token ids in every row."""
        mask = torch.zeros(
            self.batch_size, VOCAB_SIZE, dtype=torch.bool, device=self.device
        )
        mask[:, list(token_ids)] = True
        return mask

    @torch.no_grad()
    def sample(self, allowed, temperature=1.0):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        mask = allowed if torch.is_tensor(allowed) else self.allowed_mask(allowed)
        logits = self.world_model.obs_head(self.hidden).float() / temperature
        logits = logits.masked_fill(~mask, float("-inf"))
        tokens = torch.multinomial(logits.softmax(-1), 1).squeeze(1)
        self.append(tokens)
        return tokens


POLICY_FEATURES = ("both", "last", "mean")


def policy_features(session, mode="both"):
    if mode == "last":
        return session.hidden
    if mode == "mean":
        return session.obs_hidden.mean(dim=1)
    if mode == "both":
        return torch.cat((session.hidden, session.obs_hidden.mean(dim=1)), dim=-1)
    raise ValueError(f"mode must be one of {POLICY_FEATURES}, not {mode!r}")


def policy_input_dim(embed_dim, mode="both"):
    return 2 * embed_dim if mode == "both" else embed_dim


class RealCollector:
    def __init__(
        self,
        world_model,
        policy,
        ctx_len=CTX_LEN,
        warmup_steps=256,
        epsilon=0.1,
        max_episode_steps=256,
        random_action_fraction=0.1,
        feature_mode="both",
    ):
        if ctx_len < STEP_LEN or ctx_len % STEP_LEN:
            raise ValueError("ctx_len must contain whole timesteps")
        if ctx_len + OUT_LEN > world_model.max_tokens:
            raise ValueError(
                "Context plus prediction exceeds the world model's max_tokens"
            )
        if (
            warmup_steps < 0
            or not 0 <= epsilon <= 1
            or not 0 <= random_action_fraction <= 1
            or max_episode_steps < 1
        ):
            raise ValueError("Invalid collection settings")
        self.world_model, self.policy = world_model, policy
        self.ctx_len = ctx_len
        self.warmup_steps, self.epsilon = warmup_steps, epsilon
        self.max_episode_steps = max_episode_steps
        self.random_action_fraction = random_action_fraction
        self.feature_mode = feature_mode
        self.total_steps = 0
        self.city = None

    def reset(self, replay):
        self.city = random_city()
        self.history = observe(self.city)
        self.episode_steps = 0
        replay.start(self.history)

    @torch.no_grad()
    def collect(self, replay, num_steps):
        model_mode, policy_mode = self.world_model.training, self.policy.training
        self.world_model.eval()
        self.policy.eval()
        session = None  # Fresh after any intervening world-model optimizer step
        try:
            for _ in range(num_steps):
                if self.city is None:
                    self.reset(replay)
                    session = None
                # Ends at obs_t, with room reserved for action_t
                context = self.history[-(self.ctx_len - 1) :]
                if (
                    self.total_steps < self.warmup_steps
                    or random.random() < self.epsilon
                ):
                    action_idx = exploration_action(
                        self.city, self.random_action_fraction
                    )
                else:
                    if session is None:
                        session = CachedContext(self.world_model, context)
                    dist, _ = self.policy(policy_features(session, self.feature_mode))
                    action_idx = dist.sample().item()

                wm_context = context + [ACTION_TOKEN_IDS[action_idx]]
                reward, terminated = self.city.step(ACTION_NAMES[action_idx])
                self.episode_steps += 1
                truncated = (
                    not terminated and self.episode_steps >= self.max_episode_steps
                )
                if not terminated:
                    self.city.env_step()
                # At termination, stop stepping grid
                next_tokens = observe(self.city)
                replay.add(action_idx, float(reward), bool(terminated), next_tokens)
                self.total_steps += 1
                if terminated or truncated:
                    self.city = None
                else:
                    self.history = (wm_context + next_tokens)[-(self.ctx_len - 1) :]
                    if session is not None:
                        session.append([ACTION_TOKEN_IDS[action_idx]] + next_tokens)
                        session.crop(self.ctx_len - 1)
        finally:
            self.world_model.train(model_mode)
            self.policy.train(policy_mode)
        return num_steps


def generate_observation(session, temperature=1.0, grid_temperature=None):
    grid_temperature = temperature if grid_temperature is None else grid_temperature
    B = session.batch_size
    shortest = min(len(row) for row in session.tokens)
    previous = (
        [row[-STEP_LEN:-1] for row in session.tokens] if shortest >= STEP_LEN else None
    )
    columns = [session.sample(TASK_INDEX_TOKEN_IDS, temperature)]
    if OBS_MODE == "compact":
        for ids in (ROW_TOKEN_IDS, COL_TOKEN_IDS, STOP_TOKEN_IDS):
            columns.append(session.sample(ids, grid_temperature))
    else:
        cells = session.allowed_mask(CELL_TOKEN_IDS)
        placed = torch.zeros(B, dtype=torch.bool, device=session.device)
        for _ in range(GRID_ROWS * GRID_COLS):
            mask = cells.clone()
            mask[placed, VOCAB["P"]] = False
            token = session.sample(mask, grid_temperature)
            placed |= token == VOCAB["P"]
            columns.append(token)
    generated = torch.stack(columns, dim=1)
    tasks = torch.tensor(
        [sample_task_token(player_position(row)) for row in generated.tolist()],
        device=session.device,
    )
    if previous is not None:
        previous = torch.tensor(previous, device=session.device)
        carried = generated[:, TASK_INDEX_POS] == previous[:, TASK_INDEX_POS]
        tasks = torch.where(carried, previous[:, TASK_POS], tasks)
    session.append(tasks)
    return torch.cat((generated, tasks[:, None]), dim=1).tolist()


class CityDataset(Dataset):
    def __init__(self, replay, steps, tile=False):
        if steps < 1:
            raise ValueError("Windows need at least one step")
        self.windows = []
        for episode in replay.episodes:
            n = len(episode)
            starts = range(0, n, steps) if tile else range(1 - steps, n)
            self.windows.extend(
                (episode, max(start, 0), min(start + steps, n), n) for start in starts
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        episode, start, end, n = self.windows[idx]
        sequence = torch.tensor(episode.tokens(start, end), dtype=torch.long)
        x, y = sequence[:-1], sequence[1:].clone()
        # Action t of the window sits at OUT_LEN + t * STEP_LEN; y[i] targets token i + 1.
        positions = OUT_LEN + STEP_LEN * torch.arange(end - start)
        y[: OUT_LEN - 1] = -1  # the first observation, nothing to condition on
        y[positions - 1] = -1  # actions generated by policy, separate from dynamics
        y[positions + TASK_POS] = -1  # task issued by environment
        rewards = torch.tensor(episode.rewards[start:end], dtype=torch.float32)
        terminated = torch.zeros(end - start, dtype=torch.long)
        terminated[-1] = int(episode.terminated and end == n == len(episode))
        return x, y, positions, rewards, terminated


def collate_world_model(batch):
    xs, ys, positions, rewards, terminated = zip(*batch)
    return (
        pad_sequence(xs, batch_first=True, padding_value=VOCAB["<PAD>"]),
        pad_sequence(ys, batch_first=True, padding_value=-1),
        pad_sequence(positions, batch_first=True, padding_value=-1),
        pad_sequence(rewards, batch_first=True),
        pad_sequence(terminated, batch_first=True),
    )


def world_model_losses(world_model, x, y, positions, rewards, terminated):
    h = world_model(x)
    logits = world_model.obs_head(h)
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1
    )
    valid = positions >= 0
    rows = torch.arange(len(x), device=x.device)[:, None].expand_as(positions)
    action_h = h[rows[valid], positions[valid]]
    rewards, terminated = rewards[valid], terminated[valid]
    reward, termination_logits = world_model.outcomes(action_h)
    scale = world_model.reward_scale
    reward_loss = F.mse_loss(
        reward.float() / scale, rewards / scale
    )  # normalize reward based on env definition
    termination_loss = F.cross_entropy(termination_logits.float(), terminated)
    return (
        token_loss + reward_loss + termination_loss,
        token_loss,
        reward_loss,
        termination_loss,
    )


def train_world_model(
    world_model,
    optimizer,
    replay,
    window_steps,
    num_updates=10,
    batch_size=2,
    scheduler=None,
    autocast=False,
):
    dataset = CityDataset(replay, window_steps)
    if not len(dataset):
        raise ValueError("Collect real transitions before training")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_world_model,
        num_workers=0,
    )
    batches = iter(loader)
    device = next(world_model.parameters()).device
    previous_mode = world_model.training
    world_model.train()
    metrics = []
    try:
        for _ in range(num_updates):
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(loader)
                batch = next(batches)
            x, y, positions, rewards, terminated = [v.to(device) for v in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=autocast and device.type == "cuda",
            ):
                loss, token_loss, reward_loss, termination_loss = world_model_losses(
                    world_model, x, y, positions, rewards, terminated
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            metrics.append(
                dict(
                    loss=loss.item(),
                    tokens=token_loss.item(),
                    reward=reward_loss.item(),
                    termination=termination_loss.item(),
                )
            )
    finally:
        optimizer.zero_grad(set_to_none=True)
        world_model.train(previous_mode)
    return metrics


@contextmanager
def frozen_world(world_model):
    modes = [(module, module.training) for module in world_model.modules()]
    flags = [(p, p.requires_grad) for p in world_model.parameters()]
    world_model.eval()
    for p, _ in flags:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode


@dataclass(frozen=True)
class ImaginedBatch:
    features: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor


def lambda_returns(rewards, values, terminals, bootstrap, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(bootstrap)
    for t in reversed(range(len(rewards))):
        next_value = bootstrap if t == len(rewards) - 1 else values[t + 1]
        continuation = (~terminals[t]).to(rewards.dtype)
        delta = rewards[t] + gamma * continuation * next_value - values[t]
        running = delta + gamma * lam * continuation * running
        advantages[t] = running
    return advantages + values, advantages


@torch.no_grad()
def imagine(
    world_model,
    policy,
    replay,
    ctx_len=CTX_LEN,
    num_rollouts=4,
    horizon=8,
    gamma=0.99,
    lam=0.95,
    temperature=1.0,
    grid_temperature=None,
    reward_scale=1.0,
    sample_termination=False,
    from_real_context=True,
    feature_mode="both",
):
    if num_rollouts < 1 or horizon < 1:
        raise ValueError("num_rollouts and horizon must be positive")
    if ctx_len % STEP_LEN or ctx_len < STEP_LEN:
        raise ValueError("ctx_len must contain whole timesteps")
    if ctx_len + OUT_LEN > world_model.max_tokens:
        raise ValueError("Insufficient max_tokens for generated observations")
    previous_mode = policy.training
    policy.eval()
    B = num_rollouts
    # history before the seed state, leaving room for its action
    history_steps = ctx_len // STEP_LEN - 1
    try:
        with frozen_world(world_model):
            chosen = replay.sample_steps(B)
            if from_real_context:
                # drop recorded action since policy chooses its own
                contexts = [e.tokens(max(t - history_steps, 0), t) for e, t in chosen]
                session = CachedContext(world_model, contexts)
            else:
                starts = []
                for e, t in chosen:
                    context = e.tokens(max(t - history_steps, 0), t, with_action=True)
                    single = CachedContext(world_model, context)
                    starts.append(
                        generate_observation(single, temperature, grid_temperature)[0]
                    )
                # strict real/generated boundary: re-encode without the seeds.
                session = CachedContext(world_model, starts)
            device = session.device
            action_token_ids = torch.tensor(ACTION_TOKEN_IDS, device=device)
            alive = torch.ones(B, dtype=torch.bool, device=device)
            features, actions, log_probs, values, rewards, terminals = (
                [] for _ in range(6)
            )
            bootstrap = torch.zeros(B, device=device)
            for step in range(horizon):
                session.crop(ctx_len - 1)
                feature = policy_features(session, feature_mode).detach().clone()
                dist, value = policy(feature)
                action = dist.sample()
                features.append(feature)
                actions.append(action)
                log_probs.append(dist.log_prob(action))
                values.append(value)
                session.append(action_token_ids[action])
                reward, termination_logits = world_model.outcomes(session.hidden)
                p_terminal = termination_logits.float().softmax(-1)[:, 1]
                terminal = (
                    torch.bernoulli(p_terminal).bool()
                    if sample_termination
                    else p_terminal > 0.5
                )
                rewards.append(reward.float())
                terminals.append(terminal)
                alive &= ~terminal
                if not alive.any():
                    break
                generate_observation(session, temperature, grid_temperature)
                session.crop(ctx_len - 1)
            else:
                # horizon cutoff: value the next generated state for live rows.
                _, bootstrap = policy(policy_features(session, feature_mode))
            steps = len(rewards)
            terminal_t = torch.stack(terminals, dim=1)  # (B, steps)
            first_terminal = torch.where(
                terminal_t.any(1),
                terminal_t.float().argmax(1) + 1,
                torch.full((B,), steps, device=device),
            )
            pieces = {name: [] for name in ImaginedBatch.__dataclass_fields__}
            stacked = dict(
                features=torch.stack(features, 1),
                actions=torch.stack(actions, 1),
                old_log_probs=torch.stack(log_probs, 1),
                old_values=torch.stack(values, 1),
                rewards=torch.stack(rewards, 1),
                terminated=terminal_t,
            )
            for b in range(B):
                n = int(first_terminal[b])
                row = {k: v[b, :n] for k, v in stacked.items()}
                returns, advantages = lambda_returns(
                    row["rewards"] / reward_scale,
                    row["old_values"],
                    row["terminated"],
                    bootstrap[b],
                    gamma,
                    lam,
                )
                row["returns"], row["advantages"] = returns, advantages
                for name in pieces:
                    pieces[name].append(row[name])
    finally:
        policy.train(previous_mode)
    return ImaginedBatch(**{name: torch.cat(v).detach() for name, v in pieces.items()})


def train_policy(
    policy,
    optimizer,
    batch,
    epochs=4,
    minibatch_size=64,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    value_clip=0.2,
):
    """PPO on detached imagined features. No replay or environment argument."""

    if not isinstance(batch, ImaginedBatch):
        raise TypeError("Policy updates require an ImaginedBatch")
    previous_mode = policy.training
    policy.train()
    advantages = batch.advantages
    # standardizing a handful of samples mostly amplifies noise.
    if len(advantages) >= 8:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
    metrics = []
    try:
        for _ in range(epochs):
            indices = torch.randperm(len(batch.actions), device=batch.actions.device)
            for start in range(0, len(indices), minibatch_size):
                mb = indices[start : start + minibatch_size]
                dist, value = policy(batch.features[mb])
                ratio = (
                    dist.log_prob(batch.actions[mb]) - batch.old_log_probs[mb]
                ).exp()
                actor_loss = -torch.minimum(
                    ratio * advantages[mb],
                    ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages[mb],
                ).mean()
                critic_loss = (value - batch.returns[mb]) ** 2
                if value_clip:
                    clipped = batch.old_values[mb] + (
                        value - batch.old_values[mb]
                    ).clamp(-value_clip, value_clip)
                    critic_loss = torch.maximum(
                        critic_loss, (clipped - batch.returns[mb]) ** 2
                    )
                critic_loss = critic_loss.mean()
                entropy = dist.entropy().mean()
                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                metrics.append(
                    dict(
                        loss=loss.item(),
                        actor=actor_loss.item(),
                        critic=critic_loss.item(),
                        entropy=entropy.item(),
                    )
                )
    finally:
        optimizer.zero_grad(set_to_none=True)
        policy.train(previous_mode)
    return metrics


@dataclass
class DynaConfig:
    context_steps: int = 24
    warmup_steps: int = 1024
    pretrain_updates: int = 100
    real_steps: int = 32
    world_updates: int = 30
    world_batch_size: int = 4
    imagined_rollouts: int = 32
    imagination_horizon: int = 16
    policy_epochs: int = 4
    policy_batch_size: int = 64
    epsilon: float = 0.3
    random_action_fraction: float = 0.1
    max_episode_steps: int = 256
    replay_capacity: int = 10_000
    world_lr: float = 3e-4
    world_weight_decay: float = 0.1
    world_warmup_updates: int = 100
    world_dropout: float = 0.1
    autocast: bool = False
    policy_lr: float = 1e-4
    reward_scale: float = 20.0
    gamma: float = 0.995
    gae_lambda: float = 0.95
    temperature: float = 1.0
    grid_temperature: float = 0.5
    sample_termination: bool = False
    imagine_from_real_context: bool = True
    policy_features: str = "both"
    clip_eps: float = 0.2
    value_coef: float = 0.5
    value_clip: float = 0.2
    entropy_coef: float = 0.02

    def __post_init__(self):
        for name in (
            "context_steps",
            "warmup_steps",
            "pretrain_updates",
            "real_steps",
            "world_updates",
            "world_batch_size",
            "imagined_rollouts",
            "imagination_horizon",
            "policy_epochs",
            "policy_batch_size",
            "max_episode_steps",
            "replay_capacity",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        probabilities = (
            self.epsilon,
            self.random_action_fraction,
            self.gamma,
            self.gae_lambda,
            self.world_dropout,
        )
        if not all(0 <= p <= 1 for p in probabilities):
            raise ValueError(
                "epsilon, random_action_fraction, gamma, gae_lambda and world_dropout must lie in [0, 1]"
            )
        positive = (
            self.world_lr,
            self.policy_lr,
            self.temperature,
            self.grid_temperature,
            self.clip_eps,
            self.reward_scale,
        )
        if min(positive) <= 0:
            raise ValueError(
                "Learning rates, temperatures, clip_eps and reward_scale must be positive"
            )
        if (
            min(
                self.value_coef,
                self.entropy_coef,
                self.value_clip,
                self.world_weight_decay,
            )
            < 0
        ):
            raise ValueError("Loss coefficients and weight decay must be non-negative")
        if self.policy_features not in POLICY_FEATURES:
            raise ValueError(f"policy_features must be one of {POLICY_FEATURES}")
        if self.world_warmup_updates < 0:
            raise ValueError("world_warmup_updates must be non-negative")


def build_models(
    config, device="cpu", model_type="gpt-nano", agent_hidden_dim=256, wm_hidden_dim=256
):
    if model_type not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model_type {model_type!r}; choose from {sorted(MODEL_PRESETS)}"
        )
    num_layers, num_heads, embed_dim = MODEL_PRESETS[model_type]
    transformer_config = TransformerConfig(
        tokens_per_block=STEP_LEN,
        max_blocks=config.context_steps + 1,
        attention="causal",
        num_layers=num_layers,
        num_heads=num_heads,
        embed_dim=embed_dim,
        embed_pdrop=config.world_dropout,
        resid_pdrop=config.world_dropout,
        attn_pdrop=config.world_dropout,
    )
    world_model = WorldModel(
        transformer_config,
        VOCAB_SIZE,
        hidden=wm_hidden_dim,
        reward_scale=config.reward_scale,
    ).to(device)
    policy = ActorCritic(
        policy_input_dim(embed_dim, config.policy_features),
        len(ACTION_NAMES),
        hidden=agent_hidden_dim,
    ).to(device)
    return world_model, policy


def world_param_groups(world_model, weight_decay):
    """Decay matrices only; biases, LayerNorm and embeddings are left alone."""
    embeddings = {id(world_model.embed.weight), id(world_model.pos_emb.weight)}
    decay, no_decay = [], []
    for p in world_model.parameters():
        if p.ndim >= 2 and id(p) not in embeddings:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        dict(params=decay, weight_decay=weight_decay),
        dict(params=no_decay, weight_decay=0.0),
    ]


class DynaTrainer:
    """Alternate real collection, supervised model learning and imagined PPO."""

    def __init__(self, world_model, policy, config):
        if set(map(id, world_model.parameters())) & set(map(id, policy.parameters())):
            raise ValueError("Policy optimizer must not own world-model parameters")
        self.world_model, self.policy, self.config = world_model, policy, config
        self.replay = Replay(config.replay_capacity)
        self.collector = RealCollector(
            world_model,
            policy,
            ctx_len=config.context_steps * STEP_LEN,
            warmup_steps=config.warmup_steps,
            epsilon=config.epsilon,
            max_episode_steps=config.max_episode_steps,
            random_action_fraction=config.random_action_fraction,
            feature_mode=config.policy_features,
        )
        self.world_optimizer = torch.optim.AdamW(
            world_param_groups(world_model, config.world_weight_decay),
            lr=config.world_lr,
            betas=(0.9, 0.95),
        )
        self.set_world_schedule(0)
        self.policy_optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.policy_lr
        )
        self.bootstrapped = False
        self.pretrain_updates = config.pretrain_updates
        self.iteration = 0

    def _world_lr_scale(self, updates):
        """Linear warmup over world_warmup_updates, then constant."""
        return min(1.0, (updates + 1) / max(self.config.world_warmup_updates, 1))

    def set_world_schedule(self, updates):
        """Schedule config.world_lr as if `updates` world-model updates already ran.

        Resuming passes the saved count, so warmup continues rather than restarts.
        """
        for group in self.world_optimizer.param_groups:
            group["lr"] = group["initial_lr"] = self.config.world_lr
        self.world_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.world_optimizer, self._world_lr_scale, last_epoch=updates - 1
        )

    def _train_world(self, num_updates):
        c = self.config
        return train_world_model(
            self.world_model,
            self.world_optimizer,
            self.replay,
            c.context_steps,
            num_updates,
            c.world_batch_size,
            scheduler=self.world_scheduler,
            autocast=c.autocast,
        )

    def bootstrap(self):
        if self.bootstrapped:
            return []
        self.collector.collect(self.replay, self.config.warmup_steps)
        metrics = self._train_world(self.pretrain_updates)
        self.bootstrapped = True
        return metrics

    def step(self):
        self.bootstrap()
        c = self.config
        self.collector.collect(self.replay, c.real_steps)
        wm_metrics = self._train_world(c.world_updates)
        imagined = imagine(
            self.world_model,
            self.policy,
            self.replay,
            ctx_len=c.context_steps * STEP_LEN,
            num_rollouts=c.imagined_rollouts,
            horizon=c.imagination_horizon,
            gamma=c.gamma,
            lam=c.gae_lambda,
            temperature=c.temperature,
            grid_temperature=c.grid_temperature,
            reward_scale=c.reward_scale,
            sample_termination=c.sample_termination,
            from_real_context=c.imagine_from_real_context,
            feature_mode=c.policy_features,
        )
        with frozen_world(self.world_model):
            pi_metrics = train_policy(
                self.policy,
                self.policy_optimizer,
                imagined,
                epochs=c.policy_epochs,
                minibatch_size=c.policy_batch_size,
                clip_eps=c.clip_eps,
                value_coef=c.value_coef,
                entropy_coef=c.entropy_coef,
                value_clip=c.value_clip,
            )
        self.iteration += 1
        return dict(
            iteration=self.iteration,
            real_steps=self.collector.total_steps,
            replay_size=len(self.replay),
            imagined_steps=len(imagined.actions),
            world_loss=wm_metrics[-1]["loss"],
            token_loss=wm_metrics[-1]["tokens"],
            reward_loss=wm_metrics[-1]["reward"],
            termination_loss=wm_metrics[-1]["termination"],
            policy_loss=pi_metrics[-1]["loss"],
            policy_entropy=pi_metrics[-1]["entropy"],
            imagined_reward=imagined.rewards.mean().item(),
            world_lr=self.world_scheduler.get_last_lr()[0],
        )
