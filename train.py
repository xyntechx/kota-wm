import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import time

import torch

from dyna import (
    MODEL_PRESETS,
    OBS_MODE,
    DynaConfig,
    DynaTrainer,
    build_models,
    set_seed,
)
from evaluate import evaluate

CHECKPOINT_NAME = "dyna_checkpoint.pt"
BEST_CHECKPOINT_NAME = "dyna_checkpoint_best.pt"
METRICS_NAME = "metrics.json"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, config, model_type, world_model, actor_critic, trainer):
    torch.save(
        {
            "config": asdict(config),
            "model_type": model_type,
            "obs_mode": OBS_MODE,
            "world_model": world_model.state_dict(),
            "actor_critic": actor_critic.state_dict(),
            "world_optimizer": trainer.world_optimizer.state_dict(),
            "policy_optimizer": trainer.policy_optimizer.state_dict(),
            "iteration": trainer.iteration,
            "world_updates": trainer.world_scheduler.last_epoch,
        },
        path,
    )


def load_checkpoint(path, world_model, actor_critic, trainer):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["obs_mode"] != OBS_MODE:
        raise ValueError(
            f"Checkpoint uses KOTA_OBS={checkpoint['obs_mode']}, not {OBS_MODE}"
        )
    world_model.load_state_dict(checkpoint["world_model"])
    actor_critic.load_state_dict(checkpoint["actor_critic"])
    trainer.world_optimizer.load_state_dict(checkpoint["world_optimizer"])
    trainer.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])

    world_updates = checkpoint.get("world_updates")
    if world_updates is None:
        world_updates = max(
            (int(s["step"]) for s in checkpoint["world_optimizer"]["state"].values()),
            default=0,
        )

    trainer.set_world_schedule(world_updates)
    for group in trainer.policy_optimizer.param_groups:
        group["lr"] = trainer.config.policy_lr

    trainer.collector.warmup_steps = 0
    trainer.pretrain_updates = 0
    trainer.iteration = checkpoint["iteration"]
    return checkpoint


def train(
    config,
    *,
    iterations,
    out_dir=".",
    model_type="gpt2",
    agent_hidden_dim=1024,
    wm_hidden_dim=1024,
    seed=3407,
    checkpoint_every=10,
    resume=None,
    device=None,
    on_checkpoint=None,
    eval_episodes=50,
    eval_transitions=256,
    eval_every=10,
):
    if iterations < 1 or checkpoint_every < 1:
        raise ValueError("iterations and checkpoint_every must be positive")
    set_seed(seed)
    device = torch.device(device) if device is not None else pick_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / CHECKPOINT_NAME
    print("Device:", device)

    world_model, actor_critic = build_models(
        config,
        device=device,
        model_type=model_type,
        agent_hidden_dim=agent_hidden_dim,
        wm_hidden_dim=wm_hidden_dim,
    )
    trainer = DynaTrainer(world_model, actor_critic, config)
    if resume is not None:
        saved = load_checkpoint(resume, world_model, actor_critic, trainer)
        saved_policy_lr = saved["policy_optimizer"]["param_groups"][0]["lr"]
        print(
            f"Resumed from {resume} at iteration {trainer.iteration}; policy lr "
            f"{saved_policy_lr:g} -> {config.policy_lr:g}, world lr {config.world_lr:g}"
        )

    def checkpoint():
        save_checkpoint(
            checkpoint_path, config, model_type, world_model, actor_critic, trainer
        )
        (out_dir / METRICS_NAME).write_text(json.dumps(results, indent=1))
        if on_checkpoint is not None:
            on_checkpoint(checkpoint_path)
        print(f"Saved {checkpoint_path} at iteration {trainer.iteration}")

    def run_eval():
        result = evaluate(
            world_model, actor_critic, config, eval_episodes, eval_transitions, seed
        )
        result["iteration"] = trainer.iteration
        result["elapsed"] = time.time() - started
        results["evals"].append(result)
        results["eval"] = result
        print("Eval:", json.dumps(result))
        if result["policy"]["mean_return"] > results["best_return"]:
            results["best_return"] = result["policy"]["mean_return"]
            results["best_iteration"] = trainer.iteration
            save_checkpoint(
                out_dir / BEST_CHECKPOINT_NAME,
                config,
                model_type,
                world_model,
                actor_critic,
                trainer,
            )
            print(
                f"New best mean return {results['best_return']:.2f}; saved {out_dir / BEST_CHECKPOINT_NAME}"
            )

    metrics = []
    results = dict(
        config=asdict(config),
        obs_mode=OBS_MODE,
        model_type=model_type,
        agent_hidden_dim=agent_hidden_dim,
        wm_hidden_dim=wm_hidden_dim,
        seed=seed,
        iterations=metrics,
        evals=[],
        eval=None,
        best_return=float("-inf"),
        best_iteration=None,
    )
    started = time.time()
    warmup_metrics = trainer.bootstrap()
    print(f"Real transitions: {len(trainer.replay)}")
    if warmup_metrics:
        print("Warmup losses:", warmup_metrics[-1])
    for i in range(1, iterations + 1):
        result = trainer.step()
        result["elapsed"] = time.time() - started
        print(result)
        metrics.append(result)
        last = i == iterations
        if eval_episodes > 0 and (i % eval_every == 0 or last):
            run_eval()
        if i % checkpoint_every == 0 and not last:
            checkpoint()
    checkpoint()
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--model-type", default="gpt2", choices=sorted(MODEL_PRESETS))
    parser.add_argument("--agent-hidden-dim", type=int, default=1024)
    parser.add_argument("--wm-hidden-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", default=None, help="checkpoint to continue from")
    parser.add_argument("--device", default=None, help="cuda, mps or cpu (auto)")
    parser.add_argument(
        "--eval-episodes", type=int, default=50, help="0 skips evaluation"
    )
    parser.add_argument("--eval-transitions", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=10)
    group = parser.add_argument_group("DynaConfig")
    for field in fields(DynaConfig):
        flag = f"--{field.name.replace('_', '-')}"
        if field.type is bool:
            group.add_argument(
                flag, action=argparse.BooleanOptionalAction, default=field.default
            )
        else:
            group.add_argument(flag, type=field.type, default=field.default)
    return parser.parse_args(argv)


def config_from_args(args):
    return DynaConfig(**{f.name: getattr(args, f.name) for f in fields(DynaConfig)})


def main(argv=None):
    args = parse_args(argv)
    train(
        config_from_args(args),
        iterations=args.iterations,
        out_dir=args.out_dir,
        model_type=args.model_type,
        agent_hidden_dim=args.agent_hidden_dim,
        wm_hidden_dim=args.wm_hidden_dim,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        device=args.device,
        eval_episodes=args.eval_episodes,
        eval_transitions=args.eval_transitions,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
