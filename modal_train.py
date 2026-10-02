"""Run train.py on a Modal GPU, checkpointing to a Modal Volume.

    modal run modal_train.py::main --iterations 500
    modal run --detach modal_train.py::main --iterations 2000 --gpu H100   # keeps running if you close the terminal
    modal run modal_train.py::main --iterations 500 --resume 20260920-1430  # continue an earlier run
    modal run modal_train.py::main --iterations 500 --resume 20260920-1430/kota_checkpoint_best.pt

DynaConfig fields are overridden with JSON, e.g.
    --config '{"warmup_steps": 1024, "world_batch_size": 4}'
Observation encoding: --obs-mode compact (row/col/light, default) or grid (16x12 cells);
vocabulary: "mode" (only that encoding's tokens, default) or "shared".
"""

import json
from pathlib import Path
import time

import modal

APP_NAME = "kota-wm"
VOLUME_NAME = "kota-wm-checkpoints"
CHECKPOINT_DIR = Path("/checkpoints")
DEFAULT_GPU = "A100-40GB"
MAX_TIMEOUT = 24 * 60 * 60  # Modal's ceiling; resume from the volume for longer runs.

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("torch==2.14.0", "numpy", "networkx", "termcolor", "einops")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source(
        "dyna", "train", "evaluate", "model_free", "models", "envs"
    )
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)

# A run near Modal's timeout checkpoints and hands off to a fresh call that resumes
# exactly (train_state.pkl), so long runs chain past the 24 h ceiling.
SEGMENT_SECONDS = MAX_TIMEOUT - 90 * 60

RUN_DEFAULTS = dict(
    iterations=100,
    until=None,  # train up to this iteration instead (used by chained segments)
    gpu=None,  # GPU for chained segments; the first call's is chosen by the caller
    segment_seconds=SEGMENT_SECONDS,
    # Every snapshot_every iterations the checkpoint is copied to
    # snapshots/iter_NNNN.pt and scored by evaluate_remote in its own container,
    # in parallel with training (0 disables). Results: evals/iter_NNNN.json.
    snapshot_every=100,
    snapshot_eval_episodes=200,
    snapshot_eval_gpu="L4",
    config={},
    obs_mode="compact",  # or "grid": 194 tokens per step instead of 5 (dyna.OBS_MODE)
    vocab="mode",  # or "shared": one vocabulary for both encodings (dyna.VOCAB_MODE)
    oracle=False,  # compact observations also carry the hidden task progress (dyna.ORACLE)
    model_type="gpt-micro",
    agent_hidden_dim=1024,
    wm_hidden_dim=1024,
    seed=3407,
    checkpoint_every=10,
    resume=None,
    eval_episodes=50,
    eval_transitions=256,
    eval_every=10,
)


# One call per container: dyna fixes KOTA_OBS/KOTA_VOCAB when first imported, so
# a reused warm container would run the next call in the previous call's mode.
@app.function(
    gpu=DEFAULT_GPU,
    volumes={str(CHECKPOINT_DIR): volume},
    timeout=MAX_TIMEOUT,
    single_use_containers=True,
)
def train_remote(run: dict):
    """One training run. `run` holds a name plus any RUN_DEFAULTS overrides."""
    import os
    import shutil

    os.environ["KOTA_OBS"] = run.get(
        "obs_mode", RUN_DEFAULTS["obs_mode"]
    )  # before importing dyna
    os.environ["KOTA_VOCAB"] = run.get("vocab", RUN_DEFAULTS["vocab"])
    os.environ["KOTA_ORACLE"] = (
        "1" if run.get("oracle", RUN_DEFAULTS["oracle"]) else "0"
    )
    import train
    import dyna
    from dyna import DynaConfig

    wanted = tuple(run.get(k, RUN_DEFAULTS[k]) for k in ("obs_mode", "vocab", "oracle"))
    if (dyna.OBS_MODE, dyna.VOCAB_MODE, dyna.ORACLE) != wanted:
        raise RuntimeError(
            f"dyna was imported as {(dyna.OBS_MODE, dyna.VOCAB_MODE, dyna.ORACLE)}, run wants {wanted}"
        )

    unknown = set(run) - set(RUN_DEFAULTS) - {"name"}
    if unknown:
        raise ValueError(f"Unknown run settings: {sorted(unknown)}")
    run = {**RUN_DEFAULTS, **run}
    # A warm container keeps the volume as it was when the container started; a
    # chained segment must see the checkpoint the previous segment just committed.
    volume.reload()
    config = DynaConfig(**run["config"])
    out_dir = CHECKPOINT_DIR / run["name"]
    resume_path = None
    if run["resume"] is not None:
        resume_path = CHECKPOINT_DIR / run["resume"]
        if not run["resume"].endswith(".pt"):
            resume_path = resume_path / train.CHECKPOINT_NAME
        if not resume_path.is_file():
            raise FileNotFoundError(
                f"No checkpoint at {resume_path} in volume {VOLUME_NAME}"
            )

    def on_checkpoint(path, iteration):
        every = run["snapshot_every"]
        if every and iteration % every == 0:
            snapshot = out_dir / "snapshots" / f"iter_{iteration:04d}.pt"
            snapshot.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, snapshot)
        volume.commit()
        if every and iteration % every == 0:
            evaluate_remote.with_options(gpu=run["snapshot_eval_gpu"]).spawn(
                str(snapshot.relative_to(CHECKPOINT_DIR)), run["snapshot_eval_episodes"]
            )

    until = run["until"]
    if until is None and resume_path is None:
        until = run["iterations"]
    print(
        f"Run {run['name']}: {run['iterations'] if until is None else f'until iteration {until}'} "
        f"of {run['model_type']}; config {config}"
    )
    results = train.train(
        config,
        iterations=None if until is not None else run["iterations"],
        until=until,
        time_limit=run["segment_seconds"],
        out_dir=out_dir,
        model_type=run["model_type"],
        agent_hidden_dim=run["agent_hidden_dim"],
        wm_hidden_dim=run["wm_hidden_dim"],
        seed=run["seed"],
        checkpoint_every=run["checkpoint_every"],
        resume=resume_path,
        on_checkpoint=on_checkpoint,
        eval_episodes=run["eval_episodes"],
        eval_transitions=run["eval_transitions"],
        eval_every=run["eval_every"],
    )
    results["name"] = run["name"]
    if "stopped_at" in results:
        # Resume from this run's last checkpoint (and train_state.pkl) in a new call.
        target = results["until"]
        fn = (
            train_remote
            if run["gpu"] is None
            else train_remote.with_options(gpu=run["gpu"])
        )
        call = fn.spawn({**run, "resume": run["name"], "until": target})
        results["continued_in"] = call.object_id
        print(
            f"Stopped at iteration {results['stopped_at']}; continuing to {target} in call {call.object_id}"
        )
    return results


@app.function(
    gpu="L4",
    volumes={str(CHECKPOINT_DIR): volume},
    timeout=6 * 60 * 60,
    single_use_containers=True,
)
def evaluate_remote(snapshot: str, episodes: int = 200, seed: int = 0):
    """Score one checkpoint in the real environment; writes <run>/evals/<name>.json."""
    import os

    import torch

    volume.reload()
    path = CHECKPOINT_DIR / snapshot
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    os.environ["KOTA_OBS"] = checkpoint["obs_mode"]  # before importing dyna
    os.environ["KOTA_VOCAB"] = checkpoint.get("vocab", "shared")
    os.environ["KOTA_ORACLE"] = "1" if checkpoint.get("oracle", False) else "0"
    from evaluate import evaluate, load_models

    world_model, policy, config = load_models(path, device="cuda")
    result = evaluate(world_model, policy, config, episodes, 256, seed)
    result.update(
        iteration=checkpoint["iteration"],
        snapshot=snapshot,
        episodes=episodes,
        seed=seed,
    )
    out = path.parent.parent / "evals" / (path.stem + ".json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    volume.commit()
    print(
        f"{snapshot}: return {result['policy']['mean_return']:.1f} "
        f"completion {result['policy']['completion_rate']:.3f}"
    )
    return result


def _download(run_name):
    remote = f"{run_name}/kota_checkpoint.pt"
    local = Path(__file__).parent / "checkpoints" / run_name / "kota_checkpoint.pt"
    local.parent.mkdir(parents=True, exist_ok=True)
    with local.open("wb") as f:
        for chunk in volume.read_file(remote):
            f.write(chunk)
    print(f"Downloaded {remote} -> {local}")


def _summary_row(results):
    iterations = results["iterations"]
    ev = results["eval"] or {}
    policy, wm = ev.get("policy", {}), ev.get("world_model", {})
    return dict(
        name=results["name"],
        iterations=len(iterations),
        best_iteration=results.get("best_iteration"),
        sec_per_iter=(
            iterations[-1]["elapsed"] / len(iterations) if iterations else None
        ),
        mean_return=policy.get("mean_return"),
        sem_return=policy.get("sem_return"),
        best_return=results.get("best_return"),
        mean_length=policy.get("mean_length"),
        completion_rate=policy.get("completion_rate"),
        random_return=ev.get("random", {}).get("mean_return"),
        wm_token_acc=wm.get("token_accuracy"),
        wm_reward_acc=wm.get("reward_accuracy"),
        wm_reward_mse=wm.get("reward_mse"),
        wm_term_acc=wm.get("termination_accuracy"),
        final_token_loss=iterations[-1]["token_loss"] if iterations else None,
    )


def _print_table(rows):
    rows = sorted(
        rows,
        key=lambda r: -(
            r["mean_return"] if r["mean_return"] is not None else float("-inf")
        ),
    )
    columns = list(rows[0])
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    widths = [max(len(c), *(len(fmt(r[c])) for r in rows)) for c in columns]
    print("  ".join(c.ljust(w) for c, w in zip(columns, widths)))
    for r in rows:
        print("  ".join(fmt(r[c]).ljust(w) for c, w in zip(columns, widths)))


@app.local_entrypoint()
def main(
    iterations: int = 100,
    run_name: str = "",
    config: str = "{}",
    obs_mode: str = "compact",
    model_type: str = "gpt-micro",
    agent_hidden_dim: int = 1024,
    wm_hidden_dim: int = 1024,
    seed: int = 3407,
    checkpoint_every: int = 10,
    eval_episodes: int = 50,
    eval_every: int = 10,
    resume: str = "",
    gpu: str = DEFAULT_GPU,
    download: bool = False,
):
    run = dict(
        name=run_name or time.strftime("%Y%m%d-%H%M"),
        iterations=iterations,
        config=json.loads(config),
        obs_mode=obs_mode,
        model_type=model_type,
        agent_hidden_dim=agent_hidden_dim,
        wm_hidden_dim=wm_hidden_dim,
        seed=seed,
        checkpoint_every=checkpoint_every,
        eval_episodes=eval_episodes,
        eval_every=eval_every,
        resume=resume or None,
        gpu=gpu,
    )
    fn = train_remote if gpu == DEFAULT_GPU else train_remote.with_options(gpu=gpu)
    print(
        f"Starting run {run['name']!r} on {gpu}; checkpoints -> {VOLUME_NAME}:/{run['name']}/"
    )
    results = fn.remote(run)
    _print_table([_summary_row(results)])
    if download:
        _download(run["name"])
    else:
        print(
            f"Fetch it with: modal volume get {VOLUME_NAME} {run['name']}/kota_checkpoint.pt"
        )


@app.function(
    gpu="L4",
    cpu=4,
    volumes={str(CHECKPOINT_DIR): volume},
    timeout=MAX_TIMEOUT,
    single_use_containers=True,
)
def train_model_free_remote(run: dict):
    """One model-free run (compact obs, mode vocab); outputs under <name>/ in the volume.

    `run`: name, seed, total_steps, eval_every, eval_episodes, algo ("ppo" or "dqn",
    model_free.ALGOS) and config (ModelFreeConfig or DQNConfig fields).
    """
    import os

    os.environ["KOTA_OBS"], os.environ["KOTA_VOCAB"], os.environ["KOTA_ORACLE"] = (
        "compact",
        "mode",
        "0",
    )
    import model_free

    volume.reload()
    algo = run.get("algo", "ppo")
    config_class, train, *_ = model_free.ALGOS[algo]
    config = config_class(**run.get("config", {}))
    results = train(
        config,
        run["total_steps"],
        CHECKPOINT_DIR / run["name"],
        seed=run.get("seed", 3),
        eval_episodes=run.get("eval_episodes", 200),
        eval_every=run.get("eval_every", 500_000),
        on_eval=volume.commit,
    )
    volume.commit()
    evals = results["evals"]
    return dict(
        name=run["name"],
        best_return=results["best_return"],
        best_steps=results["best_steps"],
        final=evals[-1] if evals else None,
    )
