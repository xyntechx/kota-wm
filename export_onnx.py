"""Export a kota-wm checkpoint's world model and policy as one ONNX step graph for the website.

    python export_onnx.py <checkpoint.pt> --output ../Website/public/models/kota-wm.onnx --verify
"""
import argparse
import os
from pathlib import Path
import random
import sys

import torch

KOTA_WM = Path(__file__).resolve().parents[2] / "kota-wm"


class OnnxStep(torch.nn.Module):
    def __init__(self, world_model, policy, out_len, feature_mode, outcome_mode="action"):
        super().__init__()
        self.wm, self.policy = world_model, policy
        self.out_len, self.feature_mode = out_len, feature_mode
        self.outcome_mode = outcome_mode
        config = world_model.config
        self.n_layer, self.n_head = config.num_layers, config.num_heads
        self.head_size = config.embed_dim // config.num_heads

    def forward(self, ids, obs_hidden_in, *past):
        wm = self.wm
        B, T = ids.shape
        L = past[0].shape[2]
        positions = torch.arange(T, device=ids.device) + L
        x = wm.embed(ids) + wm.pos_emb(positions)[None]
        # Query i (absolute L + i) sees keys 0..L + i.
        keys = torch.arange(L + T, device=ids.device)[None, :]
        allowed = keys <= positions[:, None]
        bias = torch.zeros_like(allowed, dtype=x.dtype).masked_fill(~allowed, float("-inf"))
        present = []
        for i, block in enumerate(wm.backbone.blocks):
            attn = block.attn
            h = block.ln1(x)
            split = lambda t: t.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            q, k, v = split(attn.query(h)), split(attn.key(h)), split(attn.value(h))
            k = torch.cat((past[2 * i], k), dim=2)
            v = torch.cat((past[2 * i + 1], v), dim=2)
            present += [k, v]
            scores = q @ k.transpose(-2, -1) / self.head_size**0.5 + bias
            y = scores.softmax(-1) @ v
            y = y.transpose(1, 2).reshape(B, T, self.n_head * self.head_size)
            x = x + attn.proj(y)
            x = x + block.mlp(block.ln2(x))
        hidden = wm.backbone.ln_f(x)
        last = hidden[:, -1, :]
        obs_hidden = torch.cat((obs_hidden_in, hidden), dim=1)[:, -self.out_len :, :]
        logits = wm.obs_head(last)
        outcome_in = last
        if self.outcome_mode != "action":
            # dyna.step_outcomes: also the mean hidden state of the out_len tokens
            # before the last one (the observation an appended action follows).
            before = torch.cat((obs_hidden_in, hidden), dim=1)[:, -(self.out_len + 1) : -1, :]
            before = before.flatten(1) if self.outcome_mode == "action+obs-all" else before.mean(dim=1)
            outcome_in = torch.cat((last, before), dim=-1)
        reward, termination_logits = wm.outcomes(outcome_in)
        termination = termination_logits.softmax(-1)[:, 1]
        features = {
            "last": last,
            "mean": obs_hidden.mean(dim=1),
            "both": torch.cat((last, obs_hidden.mean(dim=1)), dim=-1),
        }[self.feature_mode]
        body = self.policy.body(features)
        policy = self.policy.pi_head(body).softmax(-1)
        value = self.policy.v_head(body).squeeze(-1)
        return (logits, reward, termination, policy, value, obs_hidden, *present)


def names(n_layer, prefix):
    return [f"{prefix}_{kv}_{i}" for i in range(n_layer) for kv in ("k", "v")]


def export(step, path):
    n_layer, n_head, head_size = step.n_layer, step.n_head, step.head_size
    ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    obs_hidden = torch.zeros(1, step.out_len, n_head * head_size)
    past = [torch.zeros(1, n_head, 4, head_size) for _ in range(2 * n_layer)]
    past_names, present_names = names(n_layer, "past"), names(n_layer, "present")
    dynamic_axes = {"ids": {1: "new"}}
    dynamic_axes.update({n: {2: "past"} for n in past_names})
    dynamic_axes.update({n: {2: "total"} for n in present_names})
    torch.onnx.export(
        step,
        (ids, obs_hidden, *past),
        path,
        input_names=["ids", "obs_hidden_in", *past_names],
        output_names=[
            "logits", "reward", "termination", "policy", "value", "obs_hidden", *present_names,
        ],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )


def verify(world_model, policy, path, dyna, config, steps=600):
    """Rollouts through dyna.CachedContext and the ONNX graph, compared at every call.

    Long enough to cross the context crop, which re-encodes from position 0.
    """
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    wm = world_model.config
    head_size = wm.embed_dim // wm.num_heads
    empty = [np.zeros((1, wm.num_heads, 0, head_size), dtype=np.float32)] * (2 * wm.num_layers)
    no_obs = np.zeros((1, dyna.OUT_LEN, wm.embed_dim), dtype=np.float32)
    past_names = names(wm.num_layers, "past")
    ctx_len = config.context_steps * dyna.STEP_LEN
    worst = dict(logits=0.0, reward=0.0, termination=0.0, policy=0.0, value=0.0)
    feeds, obs_hidden, ref, city, crops = empty, no_obs, None, None, 0

    def track(name, got, want):
        worst[name] = max(worst[name], float(np.abs(np.asarray(got) - np.asarray(want)).max()))

    def run(ids, outcomes=None):
        """`outcomes`: PyTorch reward and termination logits for an appended action."""
        nonlocal feeds, obs_hidden
        out = session.run(
            None,
            {
                "ids": np.array([ids], dtype=np.int64),
                "obs_hidden_in": obs_hidden,
                **dict(zip(past_names, feeds)),
            },
        )
        obs_hidden, feeds = out[5], out[6:]
        if outcomes is None and config.outcome_features == "action":
            outcomes = world_model.outcomes(ref.hidden)
        dist, value = policy(dyna.policy_features(ref, config.policy_features))
        track("logits", out[0], world_model.obs_head(ref.hidden).numpy())
        if outcomes is not None:  # reward/termination only mean something after an action
            reward, termination = outcomes
            track("reward", out[1], reward.numpy())
            track("termination", out[2], termination.softmax(-1)[:, 1].numpy())
        track("policy", out[3], dist.probs.numpy())
        track("value", out[4], value.numpy())

    for _ in range(steps):
        if city is None:
            city = dyna.random_city()
            ref = dyna.CachedContext(world_model, dyna.observe(city))
            feeds, obs_hidden = empty, no_obs
            run(ref.tokens[0])
        # Mostly the task's own directions: episodes long enough to reach the crop.
        if random.random() < 0.9:
            action = dyna.ACTION_NAMES.index(city._task_directions[city._dir_idx])
        else:
            action = dyna.exploration_action(city, 0.1)
        outcomes = dyna.step_outcomes(ref, [dyna.ACTION_TOKEN_IDS[action]])
        run([dyna.ACTION_TOKEN_IDS[action]], outcomes)
        _, terminated = city.step(dyna.ACTION_NAMES[action])
        if terminated:
            city = None
            continue
        city.env_step()
        observation = dyna.observe(city)
        ref.append(observation)
        run(observation)
        if ref.length > ctx_len - 1:
            ref.crop(ctx_len - 1)
            feeds, obs_hidden, crops = empty, no_obs, crops + 1
            run(ref.tokens[0])
    print(f"max abs difference vs PyTorch ({crops} crops):", worst)
    if max(worst.values()) > 1e-3:
        raise SystemExit("ONNX output diverges from PyTorch")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("public/models/kota-wm.onnx"))
    parser.add_argument("--kota-wm", type=Path, default=KOTA_WM, help="kota-wm checkout")
    parser.add_argument("--verify", action="store_true", help="compare a rollout with PyTorch")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    # dyna.py fixes OUT_LEN at import time from KOTA_OBS.
    os.environ["KOTA_OBS"] = checkpoint["obs_mode"]
    os.environ["KOTA_VOCAB"] = checkpoint.get("vocab", "shared")
    os.environ["KOTA_ORACLE"] = "1" if checkpoint.get("oracle", False) else "0"
    sys.path.insert(0, str(args.kota_wm.resolve()))
    import dyna
    from evaluate import load_models

    world_model, policy, config = load_models(args.checkpoint)
    if config.policy_features not in ("both", "last", "mean"):
        raise ValueError("ONNX export supports only world-model policy features (both, last, mean)")
    world_model.requires_grad_(False)
    policy.requires_grad_(False)
    step = OnnxStep(
        world_model, policy, dyna.OUT_LEN, config.policy_features, config.outcome_features
    ).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export(step, args.output)
    wm = world_model.config
    print(
        f"{args.output}: {checkpoint['model_type']} ({wm.num_layers} layers, {wm.num_heads} heads, "
        f"{wm.embed_dim} dims), {wm.max_tokens} tokens, context_steps {config.context_steps}, "
        f"vocab {dyna.VOCAB_SIZE}, OUT_LEN {dyna.OUT_LEN}, policy features {config.policy_features!r}, "
        f"outcome features {config.outcome_features!r}, "
        f"{args.output.stat().st_size / 1e6:.1f} MB"
    )
    if args.verify:
        random.seed(0)
        with torch.no_grad():
            verify(world_model, policy, args.output, dyna, config)


if __name__ == "__main__":
    main()
