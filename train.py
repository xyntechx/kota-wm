"""Train the Dyna world model and policy (defaults: micro-horizon8, i.e. compact-vocab-wm-actobs with gpt-micro and horizon 8).

    python train.py --until 2100 --out-dir runs/my-run
"""
import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import pickle
import random
import time

import numpy as np
import torch

from dyna import (
    MODEL_PRESETS,
    OBS_MODE,
    ORACLE,
    VOCAB_MODE,
    DynaConfig,
    DynaTrainer,
    build_models,
    set_seed,
)
from evaluate import evaluate

CHECKPOINT_NAME = "dyna_checkpoint.pt"
BEST_CHECKPOINT_NAME = "dyna_checkpoint_best.pt"
METRICS_NAME = "metrics.json"
# Everything besides the weights that a resume needs to continue exactly where the
# run stopped: replay, the episode in progress, RNG states and metrics so far.
STATE_NAME = "train_state.pkl"


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
            "vocab": VOCAB_MODE,
            "oracle": ORACLE,
            "world_model": world_model.state_dict(),
            "actor_critic": actor_critic.state_dict(),
            "world_optimizer": trainer.world_optimizer.state_dict(),
            "policy_optimizer": trainer.policy_optimizer.state_dict(),
            "iteration": trainer.iteration,
            "world_updates": trainer.world_scheduler.last_epoch,
        },
        path,
    )


def save_state(path, trainer, results):
    collector = trainer.collector
    state = dict(
        iteration=trainer.iteration,
        replay=trainer.replay,
        collector=dict(
            city=collector.city,
            history=getattr(collector, "history", None),
            episode_steps=getattr(collector, "episode_steps", 0),
            total_steps=collector.total_steps,
        ),
        rng=dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        ),
        results=results,
    )
    tmp = Path(path).with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(state, f)
    tmp.replace(path)


def load_state(path, trainer):
    """Restore save_state output onto a trainer already loaded from the same checkpoint."""
    with Path(path).open("rb") as f:
        state = pickle.load(f)
    if state["iteration"] != trainer.iteration:
        raise ValueError(
            f"{path} is from iteration {state['iteration']}, the checkpoint from {trainer.iteration}"
        )
    trainer.replay = state["replay"]
    collector = trainer.collector
    for name, value in state["collector"].items():
        setattr(collector, name, value)
    trainer.bootstrapped = True
    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return state["results"]


def load_checkpoint(path, world_model, actor_critic, trainer):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["obs_mode"] != OBS_MODE:
        raise ValueError(
            f"Checkpoint uses KOTA_OBS={checkpoint['obs_mode']}, not {OBS_MODE}"
        )
    if checkpoint.get("oracle", False) != ORACLE:
        raise ValueError(f"Checkpoint uses KOTA_ORACLE={int(checkpoint.get('oracle', False))}, not {int(ORACLE)}")
    if checkpoint.get("vocab", "shared") != VOCAB_MODE:
        raise ValueError(
            f"Checkpoint uses KOTA_VOCAB={checkpoint.get('vocab', 'shared')}, not {VOCAB_MODE}"
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
    iterations=None,
    until=None,
    time_limit=None,
    out_dir=".",
    model_type="gpt-micro",
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
    """Train for `iterations` more iterations, or until iteration `until`.

    Resuming from a run's last checkpoint also restores its train_state.pkl when
    present, so the run continues exactly (same replay, RNG and metrics). With
    `time_limit` (seconds) the run checkpoints and returns early with
    results["stopped_at"] set once that much time has passed.
    """
    if (iterations is None) == (until is None):
        raise ValueError("Pass exactly one of iterations and until")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    set_seed(seed)
    device = torch.device(device) if device is not None else pick_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / CHECKPOINT_NAME
    print("Device:", device)
    if device.type == "cuda":
        # TF32 matmuls: about 2x faster on Ampere+ at negligible precision cost here.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    world_model, actor_critic = build_models(
        config,
        device=device,
        model_type=model_type,
        agent_hidden_dim=agent_hidden_dim,
        wm_hidden_dim=wm_hidden_dim,
    )
    trainer = DynaTrainer(world_model, actor_critic, config)
    restored = None
    if resume is not None:
        saved = load_checkpoint(resume, world_model, actor_critic, trainer)
        saved_policy_lr = saved["policy_optimizer"]["param_groups"][0]["lr"]
        print(
            f"Resumed from {resume} at iteration {trainer.iteration}; policy lr "
            f"{saved_policy_lr:g} -> {config.policy_lr:g}, world lr {config.world_lr:g}"
        )
        state_path = Path(resume).with_name(STATE_NAME)
        if Path(resume).name == CHECKPOINT_NAME and state_path.is_file():
            restored = load_state(state_path, trainer)
            print(f"Restored replay ({len(trainer.replay)} steps), RNG and metrics from {state_path}")
    if until is not None:
        iterations = until - trainer.iteration
    if iterations < 1:
        raise ValueError(f"Nothing to train: iteration {trainer.iteration}, {iterations} to go")

    def checkpoint():
        save_checkpoint(
            checkpoint_path, config, model_type, world_model, actor_critic, trainer
        )
        (out_dir / METRICS_NAME).write_text(json.dumps(results, indent=1))
        save_state(out_dir / STATE_NAME, trainer, results)
        if on_checkpoint is not None:
            on_checkpoint(checkpoint_path, trainer.iteration)
        print(f"Saved {checkpoint_path} at iteration {trainer.iteration}")

    def run_eval():
        result = evaluate(
            world_model, actor_critic, config, eval_episodes, eval_transitions, seed
        )
        result["iteration"] = trainer.iteration
        result["elapsed"] = time.time() - started + elapsed_before
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
        vocab=VOCAB_MODE,
        oracle=ORACLE,
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
    if restored is not None:
        results.update(
            {k: restored[k] for k in ("iterations", "evals", "eval", "best_return", "best_iteration")}
        )
        results.pop("stopped_at", None)
        metrics = results["iterations"]
    results["until"] = trainer.iteration + iterations
    # Elapsed times continue across resumed segments.
    elapsed_before = metrics[-1]["elapsed"] if restored is not None and metrics else 0.0
    started = time.time()
    warmup_metrics = trainer.bootstrap()
    print(f"Real transitions: {len(trainer.replay)}")
    if warmup_metrics:
        print("Warmup losses:", warmup_metrics[-1])
    for i in range(1, iterations + 1):
        result = trainer.step()
        result["elapsed"] = time.time() - started + elapsed_before
        print(result)
        metrics.append(result)
        last = i == iterations
        # Schedules follow the global iteration, so resumed segments stay aligned.
        if eval_episodes > 0 and (trainer.iteration % eval_every == 0 or last):
            run_eval()
        if time_limit is not None and not last and time.time() - started > time_limit:
            results["stopped_at"] = trainer.iteration
            print(f"Time limit reached at iteration {trainer.iteration}")
            break
        if trainer.iteration % checkpoint_every == 0 and not last:
            checkpoint()
    checkpoint()
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--until", type=int, default=None, help="train up to this iteration (overrides --iterations)"
    )
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--model-type", default="gpt-micro", choices=sorted(MODEL_PRESETS))
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
        iterations=None if args.until is not None else args.iterations,
        until=args.until,
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
