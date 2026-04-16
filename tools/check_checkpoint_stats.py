#!/usr/bin/env python3
import argparse
import fnmatch
from typing import Dict, List, Tuple

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print distribution stats for specified checkpoint parameters."
    )
    parser.add_argument("checkpoint", help="Path to checkpoint (.pth)")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help=(
            "Parameter name pattern in state_dict. "
            "Supports wildcard, e.g. 'pts_bbox_head.depth_net.*running_var'. "
            "Can be used multiple times."
        ),
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional prefix filter before pattern matching.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=200,
        help="Maximum number of matched parameters to print.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["name", "max_abs", "std"],
        default="name",
        help="Sort output by parameter name or magnitude.",
    )
    parser.add_argument(
        "--only-nonfinite",
        action="store_true",
        help="Only print tensors that contain NaN/Inf.",
    )
    parser.add_argument(
        "--scan-optimizer",
        action="store_true",
        help="Also scan optimizer state (if exists).",
    )
    return parser.parse_args()


def load_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError("Unsupported checkpoint format.")


def match_name(name, patterns):
    if not patterns:
        return True
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def summarize_tensor(t: torch.Tensor):
    if not torch.is_tensor(t):
        return None
    if not t.dtype.is_floating_point:
        return None

    numel = t.numel()
    nan_cnt = int(torch.isnan(t).sum().item())
    inf_cnt = int(torch.isinf(t).sum().item())
    finite_mask = torch.isfinite(t)
    finite_cnt = int(finite_mask.sum().item())

    if finite_cnt > 0:
        finite_t = t[finite_mask]
        mean = float(finite_t.mean().item())
        std = float(finite_t.std(unbiased=False).item())
        rms = float(torch.sqrt((finite_t * finite_t).mean()).item())
        t_min = float(finite_t.min().item())
        t_max = float(finite_t.max().item())
        max_abs = float(finite_t.abs().max().item())
    else:
        mean = float("nan")
        std = float("nan")
        rms = float("nan")
        t_min = float("nan")
        t_max = float("nan")
        max_abs = float("nan")

    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "numel": numel,
        "nan_cnt": nan_cnt,
        "inf_cnt": inf_cnt,
        "finite_cnt": finite_cnt,
        "mean": mean,
        "std": std,
        "rms": rms,
        "min": t_min,
        "max": t_max,
        "max_abs": max_abs,
    }


def should_keep(stats: Dict, only_nonfinite: bool) -> bool:
    if not only_nonfinite:
        return True
    return (stats["nan_cnt"] > 0) or (stats["inf_cnt"] > 0)


def sort_key(item: Tuple[str, Dict], sort_by: str):
    name, stats = item
    if sort_by == "name":
        return name
    if sort_by == "max_abs":
        v = stats["max_abs"]
        return v if v == v else float("-inf")
    # sort_by == "std"
    v = stats["std"]
    return v if v == v else float("-inf")


def scan_state_dict(
    state_dict: Dict,
    patterns: List[str],
    prefix: str,
    only_nonfinite: bool,
):
    checked_float = 0
    kept_rows = []
    offending = 0

    for name, value in state_dict.items():
        if prefix and not name.startswith(prefix):
            continue
        if not match_name(name, patterns):
            continue

        stats = summarize_tensor(value)
        if stats is None:
            continue
        checked_float += 1

        is_offending = (stats["nan_cnt"] > 0) or (stats["inf_cnt"] > 0)
        if is_offending:
            offending += 1

        if should_keep(stats, only_nonfinite):
            kept_rows.append((name, stats))

    return checked_float, offending, kept_rows


def scan_optimizer_state(
    optimizer: Dict,
    only_nonfinite: bool,
):
    state = optimizer.get("state", {})
    if not isinstance(state, dict):
        return 0, 0, []

    checked_float = 0
    offending = 0
    kept_rows = []

    for param_id, entry in state.items():
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            stats = summarize_tensor(value)
            if stats is None:
                continue
            checked_float += 1
            name = f"optimizer.state[{param_id}].{key}"

            is_offending = (stats["nan_cnt"] > 0) or (stats["inf_cnt"] > 0)
            if is_offending:
                offending += 1

            if should_keep(stats, only_nonfinite):
                kept_rows.append((name, stats))

    return checked_float, offending, kept_rows


def print_rows(rows: List[Tuple[str, Dict]], max_print: int):
    for i, (name, s) in enumerate(rows):
        if i >= max_print:
            print(f"... truncated, showing first {max_print} entries")
            break
        print(f"\n[{i+1:04d}] {name}")
        print(f"shape={s['shape']}, dtype={s['dtype']}, numel={s['numel']}")
        print(
            f"nan={s['nan_cnt']}, inf={s['inf_cnt']}, finite={s['finite_cnt']}/{s['numel']}"
        )
        print(
            f"mean={s['mean']:.6e}, std={s['std']:.6e}, rms={s['rms']:.6e}, "
            f"min={s['min']:.6e}, max={s['max']:.6e}, max_abs={s['max_abs']:.6e}"
        )


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = load_state_dict(args.checkpoint)

    s_checked, s_offending, s_rows = scan_state_dict(
        state_dict=state_dict,
        patterns=args.param,
        prefix=args.prefix,
        only_nonfinite=args.only_nonfinite,
    )
    s_rows.sort(
        key=lambda x: sort_key(x, args.sort_by), reverse=(args.sort_by != "name")
    )

    print("=== state_dict parameter stats ===")
    print(f"checkpoint: {args.checkpoint}")
    print(f"matched float params: {s_checked}")
    print(f"offending (NaN/Inf): {s_offending}")
    print(
        f"rows printed mode: {'only non-finite' if args.only_nonfinite else 'all matched'}"
    )
    if s_rows:
        print_rows(s_rows, args.max_print)
    else:
        print("No matched floating-point parameters.")

    if args.scan_optimizer:
        if not isinstance(ckpt, dict) or "optimizer" not in ckpt:
            print("\n=== optimizer stats ===")
            print("No optimizer found in checkpoint.")
            return

        o_checked, o_offending, o_rows = scan_optimizer_state(
            optimizer=ckpt["optimizer"],
            only_nonfinite=args.only_nonfinite,
        )
        o_rows.sort(
            key=lambda x: sort_key(x, args.sort_by), reverse=(args.sort_by != "name")
        )

        print("\n=== optimizer stats ===")
        print(f"matched float params: {o_checked}")
        print(f"offending (NaN/Inf): {o_offending}")
        if o_rows:
            print_rows(o_rows, args.max_print)
        else:
            print("No matched floating-point optimizer tensors.")


if __name__ == "__main__":
    main()
