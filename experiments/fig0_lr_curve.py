import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import math

def warmup_cosine_lr(total_steps: int,
                     max_lr: float,
                     warmup_pct: float,
                     min_lr_ratio: float):
    """
    Generate LR schedule with linear warmup then cosine decay.
    - Warmup: linear from ~0 -> max_lr over warmup_steps.
    - Cosine-decay: from max_lr -> min_lr with half cosine.
    """
    steps = np.arange(total_steps, dtype=float)
    warmup_steps = int(round(total_steps * warmup_pct))
    min_lr = max_lr * min_lr_ratio
    lrs = np.empty(total_steps, dtype=float)

    if warmup_steps > 0:
        # linear warmup (start slightly above 0 to avoid exact zero)
        lrs[:warmup_steps] = max_lr * (steps[:warmup_steps] + 1) / warmup_steps
        remain = total_steps - warmup_steps
        if remain > 0:
            t = (steps[warmup_steps:] - warmup_steps) / remain  # in [0,1)
            lrs[warmup_steps:] = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(math.pi * t))
        else:
            lrs[:] = lrs[:warmup_steps][-1]
    else:
        t = steps / max(1, total_steps - 1)
        lrs = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(math.pi * t))
    return lrs

if __name__ == "__main__":
    epochs = 30
    steps_per_epoch = 10
    total_steps = epochs * steps_per_epoch

    # a) fixed lr
    lr_a = np.full(total_steps, 1e-4, dtype=float)

    # b) max_lr=1e-4, warmup_pct=0.05, min_lr_ratio=0.05
    lr_b = warmup_cosine_lr(
        total_steps=total_steps,
        max_lr=1e-4,
        warmup_pct=0.05,
        min_lr_ratio=0.05
    )

    # c) max_lr=3e-4, warmup_pct=0.10, min_lr_ratio=0.05
    lr_c = warmup_cosine_lr(
        total_steps=total_steps,
        max_lr=3e-4,
        warmup_pct=0.10,
        min_lr_ratio=0.05
    )

    x = np.arange(total_steps)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(x, lr_a, label="a) fixed_lr = 1e-4", linewidth=2)
    ax.plot(x, lr_b, label="b) max_lr=1e-4, warmup=5%, min=5%·max", linewidth=2)
    ax.plot(x, lr_c, label="c) max_lr=3e-4, warmup=10%, min=5%·max", linewidth=2)

    # Mark warmup boundaries for b and c (in step units)
    warmup_b_steps = int(round(total_steps * 0.05))
    warmup_c_steps = int(round(total_steps * 0.10))
    ax.axvline(warmup_b_steps, linestyle="--", alpha=0.5)
    ax.axvline(warmup_c_steps, linestyle="--", alpha=0.5)

    # ------ X-axis: show only epoch-end ticks; grid every 5 epochs ------
    # Ticks at epochs: 0, 5, 10, ..., 30  (positions are epoch * steps_per_epoch)
    epoch_marks = np.arange(0, epochs + 1, 5)  # [0, 5, 10, ..., 30]
    tick_positions = epoch_marks * steps_per_epoch
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(e) for e in epoch_marks])
    ax.set_xlim(0, total_steps)

    # Grid only on these major ticks along x (every 5 epochs)
    ax.grid(True, which="major", axis="x", alpha=0.6)
    ax.grid(True, which="major", axis="y", alpha=0.3)

    # Labels & title
    ax.set_title("Learning Rate Schedules (Fixed vs Warmup + Cosine-decay)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.legend()
    # fig.tight_layout()
    fig.savefig('/home/zhanghc/ucl/experiments/expr2/lr_curve.pdf')
    plt.show()
