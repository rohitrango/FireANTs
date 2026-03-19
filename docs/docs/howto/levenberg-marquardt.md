# Greedy Registration with Levenberg-Marquardt

The **Levenberg-Marquardt (LM)** optimizer is a second-order method that approximates the local Hessian of the loss to take more informed steps than plain gradient descent. It adapts a damping parameter λ at every iteration: if the loss improves, λ is shrunk (more Newton-like, larger step); if the loss worsens, λ is grown (more gradient-descent-like, smaller step). This makes the optimizer self-tuning and robust to poorly-scaled gradients.

---

## Calling Greedy with Levenberg-Marquardt (default parameters)

Pass `optimizer='levenberg'` to `GreedyRegistration`. No other changes are needed — the defaults are tuned for 3-D brain MRI registration and work well out of the box.

```python
from fireants.io.image import BatchedImages, Image
from fireants.registration.greedy import GreedyRegistration

fixed_image  = Image.load_file("fixed.nii.gz")
moving_image = Image.load_file("moving.nii.gz")

fixed_batch  = BatchedImages([fixed_image])
moving_batch = BatchedImages([moving_image])

reg = GreedyRegistration(
    scales=[8, 4, 2, 1],
    iterations=[200, 150, 100, 50],
    fixed_images=fixed_batch,
    moving_images=moving_batch,
    loss_type="fusedcc",
    loss_params={"smooth_nr": 1e-5, "smooth_dr": 1e-5},
    cc_kernel_size=7,
    optimizer="levenberg",        # <-- selects LM optimizer
    optimizer_lr=0.75,
)
reg.optimize()

moved = reg.evaluate(fixed_batch, moving_batch)
```

`optimizer_lr` is still required — it sets the overall step size multiplied onto the LM-scaled gradient.

---

## The three damping parameters

The LM optimizer has three controlling hyper-parameters. They are passed via the `optimizer_params` dictionary.

### `lambda_init` — initial damping value

| Default | `1e-2` |
|---------|--------|
| Type    | `float` or `'auto'` |
| Must be | > 0 |

λ controls the trade-off between the gradient-descent direction and the Newton direction at each voxel.

- **Large λ** → behaves like gradient descent (small, safe steps).
- **Small λ** → behaves like Newton's method (large, aggressive steps).

Setting `lambda_init='auto'` derives the starting value from the norm of the first gradient batch, which is useful when you do not know the gradient scale in advance.

### `lambda_increase_factor` — damping growth when loss increases

| Default | `1.5` |
|---------|-------|
| Type    | `float` |
| Must be | > 1.0 |

When a step causes the loss to **increase**, λ is multiplied by this factor before the next iteration. A larger value makes the optimizer retreat more aggressively to safer, shorter steps after a bad update.

### `lambda_decrease_factor` — damping shrinkage when loss decreases

| Default | `0.975` |
|---------|---------|
| Type    | `float` |
| Must be | < 1.0 |

When a step causes the loss to **decrease**, λ is multiplied by this factor, allowing the optimizer to gradually become more Newton-like as it gains confidence. Values close to 1.0 (e.g. 0.975) give a slow, stable progression; values closer to 0 (e.g. 0.7) reduce damping much faster.

### Passing custom values

```python
reg = GreedyRegistration(
    scales=[8, 4, 2, 1],
    iterations=[200, 150, 100, 50],
    fixed_images=fixed_batch,
    moving_images=moving_batch,
    loss_type="fusedcc",
    loss_params={"smooth_nr": 1e-5, "smooth_dr": 1e-5},
    cc_kernel_size=7,
    optimizer="levenberg",
    optimizer_lr=0.75,
    optimizer_params={
        "lambda_init": 0.1,           # start with stronger damping
        "lambda_increase_factor": 5.0, # retreat faster on bad steps
        "lambda_decrease_factor": 0.7, # loosen damping faster on good steps
    },
)
reg.optimize()
```

The constraints `lambda_increase_factor > 1.0` and `lambda_decrease_factor < 1.0` are enforced at construction time and will raise an `AssertionError` if violated.

!!! note "Summary of defaults"
    | Parameter | Default | Reasonable search range |
    |-----------|---------|-------------------------|
    | `lambda_init` | `1e-2` | `[1e-4, 1.0]` (log-uniform) |
    | `lambda_increase_factor` | `1.5` | `[1.5, 10.0]` |
    | `lambda_decrease_factor` | `0.975` | `[0.60, 0.999]` |

---

## Minimal reproducible evaluation script

The script below registers a list of image/segmentation pairs, records per-label Dice scores, and saves them to a `.npy` file. It requires only `fireants`, `torch`, and `numpy`.

```python
"""
evaluate_levenberg.py
---------------------
Registers fixed/moving image pairs with the Levenberg-Marquardt optimizer
and saves per-label Dice scores.

Usage:
    python evaluate_levenberg.py \
        --pairs pairs.tsv \
        --n_labels 35 \
        --output_dir ./results

pairs.tsv format (tab-separated, no header):
    fixed.nii.gz  moving.nii.gz  fixed_seg.nii.gz  moving_seg.nii.gz
"""

import argparse
import gc
import os

import numpy as np
import torch
from tqdm import tqdm

from fireants.io.image import BatchedImages, Image
from fireants.registration.greedy import GreedyRegistration


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_batch(path: str) -> BatchedImages:
    return BatchedImages([Image.load_file(path)])


def dice_scores(
    moved_seg: torch.Tensor,
    fixed_seg: torch.Tensor,
    n_labels: int,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Per-label Dice for integer label maps of shape [B, 1, ...]."""
    dices = []
    for lab in range(1, n_labels + 1):
        pred = moved_seg == lab
        gt   = fixed_seg  == lab
        intersection = (pred & gt).sum().float()
        union        = pred.sum().float() + gt.sum().float()
        dices.append((2.0 * intersection + eps) / (union + eps))
    return torch.stack(dices)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def register_pair(
    fixed_img:  BatchedImages,
    moving_img: BatchedImages,
    args: argparse.Namespace,
) -> GreedyRegistration:
    reg = GreedyRegistration(
        scales=args.scales,
        iterations=args.iterations,
        fixed_images=fixed_img,
        moving_images=moving_img,
        loss_type="fusedcc",
        loss_params={"smooth_nr": 1e-5, "smooth_dr": 1e-5},
        cc_kernel_size=args.cc_kernel_size,
        optimizer="levenberg",
        optimizer_lr=args.lr,
        optimizer_params={
            "lambda_init":            args.lambda_init,
            "lambda_increase_factor": args.lambda_increase,
            "lambda_decrease_factor": args.lambda_decrease,
        },
        max_tolerance_iters=10,
    )
    reg.optimize()
    return reg


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # parse pairs file
    pairs = []
    with open(args.pairs) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            assert len(parts) == 4, f"Expected 4 tab-separated columns, got: {line}"
            pairs.append(parts)

    print(f"Found {len(pairs)} pairs")
    os.makedirs(args.output_dir, exist_ok=True)

    all_dice = []
    running_sum, running_count = 0.0, 0

    for fixed_path, moving_path, fixed_seg_path, moving_seg_path in tqdm(pairs):
        if args.num_samples is not None and running_count >= args.num_samples:
            break

        fixed_img  = load_batch(fixed_path)
        moving_img = load_batch(moving_path)
        fixed_seg  = load_batch(fixed_seg_path)
        moving_seg = load_batch(moving_seg_path)

        # initial (pre-registration) Dice
        init_moved     = moving_seg().detach()
        fixed_seg_tens = fixed_seg().detach()
        init_dice = dice_scores(init_moved, fixed_seg_tens, args.n_labels)

        # register and evaluate
        reg       = register_pair(fixed_img, moving_img, args)
        moved_seg = reg.evaluate(fixed_img, moving_seg).detach()
        dice      = dice_scores(moved_seg, fixed_seg_tens, args.n_labels)

        running_sum   += dice.mean().item()
        running_count += 1
        print(
            f"[{running_count:>4}] init={init_dice.mean():.4f} -> "
            f"after={dice.mean():.4f}  (running avg={running_sum / running_count:.4f})"
        )
        all_dice.append(dice.cpu().numpy())

        del reg, moved_seg, fixed_seg_tens, init_moved
        torch.cuda.empty_cache()
        gc.collect()

    all_dice = np.stack(all_dice, axis=0)   # [N, n_labels]
    out_path = os.path.join(args.output_dir, "levenberg_dice_scores.npy")
    np.save(out_path, all_dice)
    print(f"\nSaved {all_dice.shape} dice array to {out_path}")
    print(f"Mean Dice over {running_count} pairs: {all_dice.mean():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Levenberg-Marquardt registration using Dice scores"
    )
    # data
    parser.add_argument(
        "--pairs", required=True,
        help="TSV file: fixed.nii.gz  moving.nii.gz  fixed_seg.nii.gz  moving_seg.nii.gz",
    )
    parser.add_argument("--n_labels", type=int, default=35,
                        help="Number of segmentation labels (1..n_labels)")
    parser.add_argument("--output_dir", required=True, help="Directory for saved results")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit evaluation to this many pairs (useful for debugging)")
    # registration
    parser.add_argument("--scales",     type=str, default="8,4,2,1")
    parser.add_argument("--iterations", type=str, default="200,150,100,50")
    parser.add_argument("--lr",         type=float, default=0.75)
    parser.add_argument("--cc_kernel_size", type=int, default=7)
    # LM hyper-parameters
    parser.add_argument("--lambda_init",     type=float, default=1e-2,
                        help="Initial damping λ (default: 1e-2)")
    parser.add_argument("--lambda_increase", type=float, default=1.5,
                        help="Multiply λ by this when loss increases (must be > 1, default: 1.5)")
    parser.add_argument("--lambda_decrease", type=float, default=0.975,
                        help="Multiply λ by this when loss decreases (must be < 1, default: 0.975)")

    args = parser.parse_args()
    args.scales     = list(map(int, args.scales.split(",")))
    args.iterations = list(map(int, args.iterations.split(",")))
    main(args)
```

### Running the script

```bash
# Register all pairs with default LM parameters
python evaluate_levenberg.py \
    --pairs pairs.tsv \
    --n_labels 35 \
    --output_dir ./results

# Override damping parameters
python evaluate_levenberg.py \
    --pairs pairs.tsv \
    --n_labels 35 \
    --output_dir ./results \
    --lambda_init 0.1 \
    --lambda_increase 5.0 \
    --lambda_decrease 0.7

# Quick smoke-test on 3 pairs
python evaluate_levenberg.py \
    --pairs pairs.tsv \
    --n_labels 35 \
    --output_dir ./results \
    --num_samples 3
```

### Loading and inspecting the results

```python
import numpy as np

dice = np.load("results/levenberg_dice_scores.npy")  # shape: [N_pairs, n_labels]
print(f"Mean Dice:         {dice.mean():.4f}")
print(f"Per-label mean:    {dice.mean(axis=0)}")
print(f"Worst label:       label {dice.mean(axis=0).argmin() + 1}  "
      f"({dice.mean(axis=0).min():.4f})")
```
