import argparse
import json

import torch
from torch.utils.data import DataLoader

from dyna import (
    ACTION_NAMES,
    ACTION_TOKEN_IDS,
    OBS_MODE,
    STEP_LEN,
    CachedContext,
    CityDataset,
    DynaConfig,
    RealCollector,
    Replay,
    build_models,
    collate_world_model,
    exploration_action,
    observe,
    policy_features,
    random_city,
    set_seed,
    world_model_losses,
)


def load_models(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["obs_mode"] != OBS_MODE:
        raise ValueError(
            f"Checkpoint was trained with KOTA_OBS={checkpoint['obs_mode']}, not {OBS_MODE}"
        )
    config = DynaConfig(**checkpoint["config"])
    world_state, policy_state = checkpoint["world_model"], checkpoint["actor_critic"]
    world_model, policy = build_models(
        config,
        device=device,
        model_type=checkpoint["model_type"],
        agent_hidden_dim=policy_state["body.0.weight"].shape[0],
        wm_hidden_dim=world_state["rew_head.0.weight"].shape[0],
    )
    world_model.load_state_dict(world_state)
    policy.load_state_dict(policy_state)
    return world_model.eval(), policy.eval(), config


def _episode_stats(returns, lengths, completed, deviated):
    returns_t = torch.tensor(returns)
    return dict(
        episodes=len(returns),
        mean_return=returns_t.mean().item(),
        std_return=returns_t.std().item() if len(returns) > 1 else 0.0,
        sem_return=(
            (returns_t.std() / len(returns) ** 0.5).item() if len(returns) > 1 else 0.0
        ),
        mean_length=sum(lengths) / len(lengths),
        tasks_completed=completed,
        tasks_deviated=deviated,
        completion_rate=completed / max(completed + deviated, 1),
    )


@torch.no_grad()
def run_episodes(config, episodes, world_model=None, policy=None):
    ctx_len = config.context_steps * STEP_LEN
    if policy is not None:
        modes = world_model.training, policy.training
        world_model.eval()
        policy.eval()
    returns, lengths, completed, deviated = [], [], 0, 0
    try:
        for _ in range(episodes):
            city = random_city()
            session = (
                CachedContext(world_model, observe(city))
                if policy is not None
                else None
            )
            total, steps = 0.0, 0
            while True:
                if session is None:
                    action_idx = exploration_action(city, config.random_action_fraction)
                else:
                    dist, _ = policy(policy_features(session, config.policy_features))
                    action_idx = dist.probs.argmax(-1).item()
                expected = city._task_directions[city._dir_idx]
                reward, terminated = city.step(ACTION_NAMES[action_idx])
                if city._need_new_task:
                    completed += ACTION_NAMES[action_idx] == expected
                    deviated += ACTION_NAMES[action_idx] != expected
                total += float(reward)
                steps += 1
                if terminated or steps >= config.max_episode_steps:
                    break
                city.env_step()
                if session is not None:
                    session.append([ACTION_TOKEN_IDS[action_idx]] + observe(city))
                    session.crop(ctx_len - 1)
            returns.append(total)
            lengths.append(steps)
    finally:
        if policy is not None:
            world_model.train(modes[0])
            policy.train(modes[1])
    return _episode_stats(returns, lengths, completed, deviated)


@torch.no_grad()
def evaluate_world_model(world_model, policy, config, transitions=256, batch_size=8):
    replay = Replay()
    collector = RealCollector(
        world_model,
        policy,
        ctx_len=config.context_steps * STEP_LEN,
        warmup_steps=transitions,  # all exploration: the policy is never queried
        epsilon=1.0,
        max_episode_steps=config.max_episode_steps,
        random_action_fraction=config.random_action_fraction,
    )
    collector.collect(replay, transitions)
    dataset = CityDataset(replay, config.context_steps, tile=True)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_world_model)
    device = next(world_model.parameters()).device
    was_training = world_model.training
    world_model.eval()
    totals = dict(tokens=0.0, reward=0.0, termination=0.0, reward_mse=0.0)
    correct = counted = steps = reward_correct = termination_correct = 0
    try:
        for batch in loader:
            x, y, positions, rewards, terminated = [v.to(device) for v in batch]
            _, token_loss, reward_loss, termination_loss = world_model_losses(
                world_model, x, y, positions, rewards, terminated
            )
            mask, valid = y != -1, positions >= 0
            totals["tokens"] += token_loss.item() * mask.sum().item()
            totals["reward"] += reward_loss.item() * valid.sum().item()
            totals["termination"] += termination_loss.item() * valid.sum().item()
            h = world_model(x)
            correct += (
                (world_model.obs_head(h).argmax(-1)[mask] == y[mask]).sum().item()
            )
            counted += mask.sum().item()
            rows = torch.arange(len(x), device=device)[:, None].expand_as(positions)
            reward, termination_logits = world_model.outcomes(
                h[rows[valid], positions[valid]]
            )
            rewards, terminated = rewards[valid], terminated[valid]
            totals["reward_mse"] += ((reward - rewards) ** 2).sum().item()
            reward_correct += (reward.round() == rewards.round()).sum().item()
            termination_correct += (
                (termination_logits.argmax(-1) == terminated).sum().item()
            )
            steps += valid.sum().item()
    finally:
        world_model.train(was_training)
    result = dict(
        tokens=totals["tokens"] / counted,
        reward=totals["reward"] / steps,
        termination=totals["termination"] / steps,
        reward_mse=totals["reward_mse"] / steps,
    )
    result["loss"] = result["tokens"] + result["reward"] + result["termination"]
    result["token_accuracy"] = correct / counted
    result["reward_accuracy"] = reward_correct / steps
    result["termination_accuracy"] = termination_correct / steps
    result["transitions"] = steps
    return result


def evaluate(world_model, policy, config, episodes=50, transitions=256, seed=0):
    set_seed(seed)
    result = dict(policy=run_episodes(config, episodes, world_model, policy))
    set_seed(seed)
    result["random"] = run_episodes(config, episodes)
    set_seed(seed)
    result["world_model"] = evaluate_world_model(
        world_model, policy, config, transitions
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--transitions", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    world_model, policy, config = load_models(args.checkpoint, device=device)
    print(
        json.dumps(
            evaluate(
                world_model, policy, config, args.episodes, args.transitions, args.seed
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
