"""End-to-end training loop: per-group optimizer + scheduler + checkpointing.

Shows the Batch-CT-hardened path: a two-param-group NAdam, a StepLR that
writes back per group, external LR edits surviving non-decay steps, and
safetensors save/load with strict resume.
Run: python examples/train_with_scheduler_and_checkpoint.py
"""

import os
import tempfile

import numpy as np

import oxideon as ox


def main() -> None:
    ox.manual_seed(0)

    # --- Tiny regression task: y = 2x + 1 + noise ---
    x_np = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(-1, 1)
    y_np = (2.0 * x_np + 1.0 + 0.05 * np.random.default_rng(0).standard_normal((64, 1))).astype(np.float32)
    x = ox.tensor(x_np)
    y = ox.tensor(y_np)

    model = ox.nn.Sequential(ox.nn.Linear(1, 8), ox.nn.Linear(8, 1))
    params = list(model.parameters())

    # --- Two param groups at different LRs, one StepLR over both ---
    opt = ox.optim.NAdam(
        [
            {"params": params[:2], "lr": 1e-2},
            {"params": params[2:], "lr": 1e-3},
        ]
    )
    sched = ox.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    print(f"base lrs: {sched.base_lrs}")
    assert sched.base_lrs == [1e-2, 1e-3]

    loss_fn = ox.nn.MSELoss()
    first, last = None, None
    for epoch in range(6):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        sched.step()
        val = float(loss.to_numpy())
        first = val if first is None else first
        last = val
        print(f"epoch {epoch}: loss = {val:.6f}, lrs = {[g['lr'] for g in opt.param_groups]}")
    assert last < first, "training did not reduce the loss"

    # --- Decay happened per group from each base LR ---
    assert opt.param_groups[0]["lr"] < 1e-2
    assert opt.param_groups[1]["lr"] < 1e-3

    # --- External LR edits survive non-decay steps (CT09) ---
    opt.param_groups[0]["lr"] = 0.123
    sched.step()  # epoch 7: not a decay step (step_size=2)
    assert opt.param_groups[0]["lr"] == 0.123, "external LR edit was wiped"

    # --- Checkpoint: save, fresh model, strict load, identical params ---
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "model.safetensors")
        ox.save(model.state_dict(), ckpt)

        fresh = ox.nn.Sequential(ox.nn.Linear(1, 8), ox.nn.Linear(8, 1))
        ox.load(ckpt, model=fresh, strict=True)
        for (k1, t1), (k2, t2) in zip(
            sorted(model.state_dict().items()), sorted(fresh.state_dict().items())
        ):
            assert k1 == k2
            assert np.array_equal(t1.to_numpy(), t2.to_numpy()), f"{k1} mismatch after load"
        print(f"checkpoint round-trip OK ({ckpt})")

    print("train_with_scheduler_and_checkpoint: OK")


if __name__ == "__main__":
    main()
