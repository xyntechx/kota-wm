"""Spawn the runs of a sweep spec on the deployed kota-wm app.

Unlike `modal run modal_train.py::sweep`, spawned calls belong to the deployed app,
so they (and their automatic handoffs past Modal's 24 h limit) keep running after
this script exits. Deploy first:

    modal deploy modal_train.py
    python sweeps/launch.py sweeps/ablations.json [run-name ...]
"""
import json
import sys

import modal

spec = json.load(open(sys.argv[1]))
only = set(sys.argv[2:])
fn = modal.Function.from_name("kota-wm", "train_remote")
defaults = spec.get("defaults", {})
for r in spec["runs"]:
    if only and r["name"] not in only:
        continue
    run = {
        **defaults,
        **r,
        "name": f"{spec['name']}/{r['name']}",
        "config": {**defaults.get("config", {}), **r.get("config", {})},
    }
    call = fn.with_options(gpu=run["gpu"]).spawn(run)
    print(run["name"], call.object_id, flush=True)
