#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Joseph Crabtree
"""Stack per-expert MoE checkpoint tensors into grouped expert parameters.

Why this exists
---------------
transformers v5 stores the routed experts of some MoE models, NemotronH
(``model_type == "nemotron_h"``) among them, as ONE parameter per layer with a
leading ``n_experts`` dimension::

    model.layers.1.mixer.experts.up_proj              # (n_experts, intermediate, hidden)

The published safetensors checkpoint keeps one ``nn.Linear`` weight per expert::

    model.layers.1.mixer.experts.0.up_proj.weight     # (intermediate, hidden)
    model.layers.1.mixer.experts.1.up_proj.weight
    ...

If the loader does not merge them, and loads with ``strict=False``, every expert
weight is skipped without any error and the expert FFNs keep their random init.
This module does the merge.

Two ways to use it
------------------
* In-process, which is what I ran originally. Call ``merge_grouped_experts(state_dict,
  target_shapes)`` on the full state dict right before it is handed to
  ``set_model_state_dict``. Get ``target_shapes`` from the live model with
  ``target_shapes_from_model``. For a DTensor, ``.shape`` is the full logical
  shape, so this works on FSDP2-sharded models too.
* Offline, through the CLI. It rewrites a safetensors checkpoint directory so the
  grouped tensors are already on disk. Run ``python mergepatch.py --help``.

The merge rule is the same in both modes and unchanged from the original patch:

1. For every target parameter ``<prefix>.experts.<proj>`` (``proj`` is one of
   gate_proj / up_proj / down_proj) that is not already in the state dict,
2. collect ``<prefix>.experts.{i}.<proj>.weight`` for ``i in range(n_experts)``,
   where ``n_experts`` is the target's leading dimension. If any key is absent,
   skip the parameter,
3. shape-check: the target's trailing dims must equal the per-expert ``(out, in)``
   shape (stack as-is) or its transpose (transpose each expert, then stack).
   Anything else is reported and not merged,
4. pop each per-expert tensor as it is stacked, so peak memory stays roughly
   flat, and store ``torch.stack(experts, dim=0)`` under the grouped name.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import shutil
import sys
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

Shape = tuple[int, ...]

DEFAULT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
DEFAULT_EXPERTS_NAME = "experts"
INDEX_FILENAME = "model.safetensors.index.json"


def grouped_regex(projections: Sequence[str] = DEFAULT_PROJECTIONS, experts_name: str = DEFAULT_EXPERTS_NAME) -> str:
    """Regex for grouped parameter names, e.g. ``...mixer.experts.up_proj``."""
    projs = "|".join(re.escape(p) for p in projections)
    return rf"^(?P<prefix>.*\.{re.escape(experts_name)})\.(?P<proj>{projs})$"


def per_expert_regex(projections: Sequence[str] = DEFAULT_PROJECTIONS, experts_name: str = DEFAULT_EXPERTS_NAME) -> str:
    """Regex for per-expert checkpoint keys, e.g. ``...mixer.experts.7.up_proj.weight``."""
    projs = "|".join(re.escape(p) for p in projections)
    return rf"^(?P<prefix>.*\.{re.escape(experts_name)})\.(?P<idx>\d+)\.(?P<proj>{projs})\.weight$"


@dataclass(frozen=True)
class MergePlan:
    """One grouped parameter and the per-expert tensors that will be stacked into it."""

    grouped_name: str
    source_keys: tuple[str, ...]
    expert_shape: Shape
    target_shape: Shape
    transpose: bool

    @property
    def num_experts(self) -> int:
        return len(self.source_keys)


@dataclass
class MergeReport:
    merged: list[MergePlan] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (grouped name, reason)
    mismatched: list[tuple[str, str]] = field(default_factory=list)  # (grouped name, reason)

    @property
    def ok(self) -> bool:
        """False if any grouped param had a shape that could not be reconciled."""
        return not self.mismatched

    @property
    def consumed_keys(self) -> set[str]:
        return {k for plan in self.merged for k in plan.source_keys}

    def summary(self) -> str:
        n_src = sum(p.num_experts for p in self.merged)
        return (
            f"merged {len(self.merged)} grouped params from {n_src} per-expert tensors; "
            f"skipped {len(self.skipped)}; shape mismatches {len(self.mismatched)}"
        )


def plan_merges(
    available_shapes: Mapping[str, Sequence[int]],
    target_shapes: Mapping[str, Sequence[int]],
    grouped_re: str | None = None,
) -> MergeReport:
    """Decide which grouped params can be built from which per-expert tensors.

    This only looks at shapes and never touches tensor data, which is what makes
    ``--dry-run`` cheap.

    Args:
        available_shapes: ``{key: shape}`` of what is in the checkpoint / state dict.
        target_shapes: ``{param_name: shape}`` the model expects. Names that do not
            look like grouped expert params are ignored.
        grouped_re: regex with named groups ``prefix`` and ``proj``. Defaults to
            ``grouped_regex()``.
    """
    pattern = re.compile(grouped_re or grouped_regex())
    report = MergeReport()
    for name, tshape in target_shapes.items():
        m = pattern.match(name)
        if not m or name in available_shapes:
            continue  # not a grouped expert param, or already present
        prefix, proj = m.group("prefix"), m.group("proj")
        tshape = tuple(int(d) for d in tshape)
        if len(tshape) != 3:
            report.mismatched.append((name, f"grouped param should be 3-D (n_experts, a, b), got {tshape}"))
            continue
        n_experts = tshape[0]
        keys = tuple(f"{prefix}.{i}.{proj}.weight" for i in range(n_experts))
        absent = [k for k in keys if k not in available_shapes]
        if absent:
            report.skipped.append((name, f"{len(absent)}/{n_experts} per-expert keys absent (first: {absent[0]})"))
            continue
        s0 = tuple(int(d) for d in available_shapes[keys[0]])
        odd = next((k for k in keys if tuple(int(d) for d in available_shapes[k]) != s0), None)
        if odd is not None:
            report.mismatched.append(
                (name, f"per-expert shapes differ: {keys[0]} {s0} vs {odd} {tuple(available_shapes[odd])}")
            )
            continue
        if tshape[1:] == s0:  # nn.Linear (out, in) stacked as-is
            transpose = False
        elif tshape[1:] == s0[::-1]:  # grouped layout stores (in, out)
            transpose = True
        else:
            report.mismatched.append((name, f"grouped {tshape} vs per-expert {s0}"))
            continue
        report.merged.append(MergePlan(name, keys, s0, tshape, transpose))
    return report


def apply_merges(state_dict: MutableMapping[str, torch.Tensor], plans: Iterable[MergePlan]) -> None:
    """Pop the per-expert tensors of each plan and store the stacked grouped tensor."""
    for plan in plans:
        parts = []
        for key in plan.source_keys:
            t = state_dict.pop(key)
            parts.append(t.t().contiguous() if plan.transpose else t)
        state_dict[plan.grouped_name] = torch.stack(parts, dim=0)
        del parts


def merge_grouped_experts(
    state_dict: MutableMapping[str, torch.Tensor],
    target_shapes: Mapping[str, Sequence[int]],
    grouped_re: str | None = None,
    dry_run: bool = False,
) -> MergeReport:
    """In-place merge of per-expert tensors in ``state_dict`` into grouped params.

    Example, inside a loader right before ``set_model_state_dict``::

        report = merge_grouped_experts(state_dict, target_shapes_from_model(model_parts))
        print(report.summary())
        if not report.ok:
            raise ValueError(report.mismatched)
    """
    shapes = {k: tuple(v.shape) for k, v in state_dict.items() if hasattr(v, "shape")}
    report = plan_merges(shapes, target_shapes, grouped_re)
    if not dry_run:
        apply_merges(state_dict, report.merged)
    return report


def target_shapes_from_model(
    model_or_parts: torch.nn.Module | Sequence[torch.nn.Module],
    strip: Sequence[str] = ("_checkpoint_wrapped_module.",),
) -> dict[str, Shape]:
    """``{param_name: full logical shape}`` for a model or list of pipeline parts.

    ``strip`` removes wrapper components (e.g. activation-checkpoint wrappers) so
    the names match the checkpoint keys, as Automodel's own loader does.
    """
    parts = model_or_parts if isinstance(model_or_parts, (list, tuple)) else [model_or_parts]
    shapes: dict[str, Shape] = {}
    for part in parts:
        for name, p in part.named_parameters():
            for s in strip:
                name = name.replace(s, "")
            shapes[name] = tuple(p.shape)
    return shapes


def infer_target_shapes(
    available_shapes: Mapping[str, Sequence[int]],
    per_expert_re: str | None = None,
    num_experts: int | None = None,
    layout: str = "stacked",
) -> dict[str, Shape]:
    """Infer grouped param names and shapes from the per-expert keys in a checkpoint.

    Use this when the live model isn't available, as in the offline CLI.
    ``layout="stacked"`` gives ``(n_experts, out, in)``, i.e. ``torch.stack`` of
    the ``nn.Linear`` weights. That is the layout of transformers'
    ``NemotronHExperts`` and of its ``MergeModulelist(dim=0)`` converter.
    ``layout="transposed"`` gives ``(n_experts, in, out)``.

    Raises ValueError if expert indices are not contiguous from 0 or if the
    count differs from ``num_experts`` when that is given.
    """
    if layout not in ("stacked", "transposed"):
        raise ValueError(f"unknown layout {layout!r}")
    pattern = re.compile(per_expert_re or per_expert_regex())
    groups: dict[tuple[str, str], dict[int, Shape]] = {}
    for key, shape in available_shapes.items():
        m = pattern.match(key)
        if not m:
            continue
        prefix, idx, proj = m.group("prefix"), int(m.group("idx")), m.group("proj")
        if key != f"{prefix}.{idx}.{proj}.weight":
            raise ValueError(f"per-expert key {key!r} is not of the form <prefix>.<idx>.<proj>.weight")
        groups.setdefault((prefix, proj), {})[idx] = tuple(int(d) for d in shape)

    targets: dict[str, Shape] = {}
    for (prefix, proj), by_idx in sorted(groups.items()):
        n = len(by_idx)
        if sorted(by_idx) != list(range(n)):
            raise ValueError(f"{prefix}.*.{proj}: expert indices are not contiguous from 0: {sorted(by_idx)[:8]} ...")
        if num_experts is not None and n != num_experts:
            raise ValueError(f"{prefix}.*.{proj}: found {n} experts, expected {num_experts}")
        s0 = by_idx[0]
        if len(s0) != 2:
            raise ValueError(f"{prefix}.0.{proj}.weight: expected a 2-D Linear weight, got {s0}")
        inner = s0 if layout == "stacked" else s0[::-1]
        targets[f"{prefix}.{proj}"] = (n, *inner)
    return targets


# --------------------------------------------------------------------------- CLI


def _parse_size(text: str) -> int:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)i?B?\s*", text, flags=re.IGNORECASE)
    if not m:
        raise argparse.ArgumentTypeError(f"bad size {text!r}; use e.g. 5GB, 500MB or a byte count")
    mult = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}[m.group(2).upper()]
    return int(float(m.group(1)) * mult)


def _parse_rename(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--rename-prefix takes OLD=NEW, e.g. backbone.=model.")
    old, new = text.split("=", 1)
    if not old:
        raise argparse.ArgumentTypeError("--rename-prefix: OLD must not be empty")
    return old, new


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mergepatch.py",
        description=(
            "Rewrite a safetensors checkpoint so per-expert MoE weights "
            "(<prefix>.experts.<i>.<proj>.weight) are stacked into grouped params "
            "(<prefix>.experts.<proj>, shape (n_experts, ...))."
        ),
    )
    p.add_argument("input_dir", type=Path, help="checkpoint directory containing *.safetensors")
    p.add_argument("output_dir", type=Path, nargs="?", help="where to write the merged checkpoint (not needed with --dry-run)")
    p.add_argument("--dry-run", action="store_true", help="report which keys would be merged; write nothing")
    p.add_argument("--num-experts", type=int, default=None, help="expected experts per layer (default: detect)")
    p.add_argument(
        "--projections",
        default=",".join(DEFAULT_PROJECTIONS),
        help="comma-separated expert projection names (default: %(default)s)",
    )
    p.add_argument("--experts-name", default=DEFAULT_EXPERTS_NAME, help="module name holding the experts (default: %(default)s)")
    p.add_argument(
        "--target-shapes",
        type=Path,
        default=None,
        help=(
            "JSON {param_name: shape} dumped from the live model (see target_shapes_from_model). "
            "When given, merges are shape-checked against it, which is how the original in-loader patch worked. "
            "Without it, shapes are inferred from the checkpoint using --layout."
        ),
    )
    p.add_argument(
        "--layout",
        choices=("stacked", "transposed"),
        default="stacked",
        help="grouped layout when inferring shapes: stacked=(E, out, in) [transformers NemotronHExperts], transposed=(E, in, out)",
    )
    p.add_argument(
        "--rename-prefix",
        type=_parse_rename,
        default=None,
        metavar="OLD=NEW",
        help="rename a key prefix before merging, e.g. backbone.=model. (the re-key I also needed)",
    )
    p.add_argument("--max-shard-size", type=_parse_size, default=_parse_size("5GB"), help="output shard size (default: 5GB)")
    p.add_argument("--overwrite", action="store_true", help="allow writing into a non-empty output_dir")
    return p


def _scan_checkpoint(files: Sequence[Path]) -> tuple[dict[str, Path], dict[str, Shape], list[str]]:
    from safetensors import safe_open

    key_to_file: dict[str, Path] = {}
    shapes: dict[str, Shape] = {}
    order: list[str] = []
    for f in files:
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                if k in key_to_file:
                    raise ValueError(f"duplicate key {k!r} in {f.name} and {key_to_file[k].name}")
                key_to_file[k] = f
                shapes[k] = tuple(h.get_slice(k).get_shape())
                order.append(k)
    return key_to_file, shapes, order


def _print_report(report: MergeReport, leftover: Sequence[str]) -> None:
    for plan in report.merged:
        first, last = plan.source_keys[0], plan.source_keys[-1]
        how = "transpose+stack" if plan.transpose else "stack"
        print(
            f"  MERGE {plan.grouped_name} {plan.target_shape} <- {plan.num_experts} x {plan.expert_shape} "
            f"[{how}] ({first} .. {last})"
        )
    for name, why in report.skipped:
        print(f"  SKIP  {name}: {why}")
    for name, why in report.mismatched:
        print(f"  SHAPE-MISMATCH {name}: {why}")
    if leftover:
        print(f"  WARNING: {len(leftover)} per-expert keys would remain unmerged, e.g. {leftover[0]}")
    print(report.summary())


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_dir: Path = args.input_dir
    files = sorted(in_dir.glob("*.safetensors"))
    if not files:
        print(f"error: no *.safetensors files in {in_dir}", file=sys.stderr)
        return 1
    if not args.dry_run:
        if args.output_dir is None:
            print("error: output_dir is required unless --dry-run", file=sys.stderr)
            return 1
        if args.output_dir.resolve() == in_dir.resolve():
            print("error: output_dir must differ from input_dir", file=sys.stderr)
            return 1
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
            print(f"error: {args.output_dir} is not empty (use --overwrite)", file=sys.stderr)
            return 1

    projections = tuple(p.strip() for p in args.projections.split(",") if p.strip())
    key_to_file, src_shapes, order = _scan_checkpoint(files)

    # Optional re-key (e.g. backbone.* -> model.*). new_name -> original key.
    src_of: dict[str, str] = {}
    for k in order:
        new = k
        if args.rename_prefix and k.startswith(args.rename_prefix[0]):
            new = args.rename_prefix[1] + k[len(args.rename_prefix[0]) :]
        if new in src_of:
            print(f"error: rename collision on {new!r}", file=sys.stderr)
            return 1
        src_of[new] = k
    shapes = {new: src_shapes[old] for new, old in src_of.items()}
    n_renamed = sum(1 for new, old in src_of.items() if new != old)

    if args.target_shapes is not None:
        target_shapes = {k: tuple(v) for k, v in json.loads(args.target_shapes.read_text()).items()}
        mode = f"target shapes from {args.target_shapes}"
    else:
        try:
            target_shapes = infer_target_shapes(
                shapes, per_expert_regex(projections, args.experts_name), args.num_experts, args.layout
            )
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        mode = f"shapes inferred from checkpoint (layout={args.layout})"
    if args.num_experts is not None and args.target_shapes is not None:
        g = re.compile(grouped_regex(projections, args.experts_name))
        bad = [n for n, s in target_shapes.items() if g.match(n) and s[0] != args.num_experts]
        if bad:
            print(f"error: {bad[0]} has {target_shapes[bad[0]][0]} experts, expected {args.num_experts}", file=sys.stderr)
            return 2

    report = plan_merges(shapes, target_shapes, grouped_regex(projections, args.experts_name))
    pe = re.compile(per_expert_regex(projections, args.experts_name))
    consumed = report.consumed_keys
    leftover = [k for k in shapes if pe.match(k) and k not in consumed]

    print(f"input: {in_dir} ({len(shapes)} tensors in {len(files)} files); {mode}")
    if n_renamed:
        print(f"renamed {n_renamed} keys: {args.rename_prefix[0]!r} -> {args.rename_prefix[1]!r}")
    _print_report(report, leftover)

    if not report.ok:
        print("error: shape mismatches; nothing written", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0
    if not report.merged and not n_renamed:
        print("nothing to merge or rename; nothing written")
        return 1

    _write_output(args, files, key_to_file, src_of, order, report)
    return 0


def _write_output(args, files, key_to_file, src_of, order, report: MergeReport) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # With --overwrite, drop shards from a previous run of this tool so the new index is the only truth.
    for old in out_dir.glob("model-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].safetensors"):
        old.unlink()

    plan_by_first = {plan.source_keys[0]: plan for plan in report.merged}
    consumed = report.consumed_keys
    new_of = {old: new for new, old in src_of.items()}

    shards: list[tuple[Path, list[str]]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    total_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        tmp = out_dir / f".shard-{len(shards):05d}.tmp"
        save_file(current, str(tmp), metadata={"format": "pt"})
        shards.append((tmp, list(current)))
        current, current_bytes = {}, 0

    with contextlib.ExitStack() as stack:
        handles = {f: stack.enter_context(safe_open(str(f), framework="pt")) for f in files}

        def load(new_key: str) -> torch.Tensor:
            old = src_of[new_key]
            return handles[key_to_file[old]].get_tensor(old)

        for old_key in order:
            key = new_of[old_key]
            if key in plan_by_first:
                plan = plan_by_first[key]
                tmp_sd = {k: load(k) for k in plan.source_keys}
                apply_merges(tmp_sd, [plan])
                name, tensor = plan.grouped_name, tmp_sd.pop(plan.grouped_name)
            elif key in consumed:
                continue  # already stacked into its grouped param
            else:
                name, tensor = key, load(key)
            nbytes = tensor.numel() * tensor.element_size()
            if current and current_bytes + nbytes > args.max_shard_size:
                flush()
            current[name] = tensor.contiguous()
            current_bytes += nbytes
            total_bytes += nbytes
        flush()

    weight_map: dict[str, str] = {}
    n = len(shards)
    for i, (tmp, names) in enumerate(shards, start=1):
        final = out_dir / f"model-{i:05d}-of-{n:05d}.safetensors"
        tmp.replace(final)
        for name in names:
            weight_map[name] = final.name
    index = {"metadata": {"total_size": total_bytes}, "weight_map": dict(sorted(weight_map.items()))}
    (out_dir / INDEX_FILENAME).write_text(json.dumps(index, indent=2) + "\n")

    # Copy everything else (config.json, tokenizer files, ...) unchanged.
    for item in sorted(args.input_dir.iterdir()):
        if item.suffix == ".safetensors" or item.name == INDEX_FILENAME:
            continue
        dest = out_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    print(f"wrote {len(weight_map)} tensors in {n} shard(s) to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
