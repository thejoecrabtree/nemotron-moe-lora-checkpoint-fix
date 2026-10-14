# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Joseph Crabtree
"""CPU-only tests for mergepatch.py using tiny fake tensors.

No GPU, no model download. Run with:  python -m pytest -q tests/
"""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mergepatch as mp  # noqa: E402

E, HIDDEN, INTER = 4, 6, 10  # experts, hidden size, expert intermediate size
LAYERS = (1, 3)  # MoE layers in a hybrid stack aren't contiguous


def fake_checkpoint(prefix="model", dtype=torch.float32):
    """Per-expert NemotronH-style tensors plus a few unrelated keys.

    Expert i of layer L has every element set to a distinct value so a
    mis-ordered or transposed stack is easy to detect.
    """
    sd = {}
    for L in LAYERS:
        base = f"{prefix}.layers.{L}.mixer.experts"
        for i in range(E):
            up = torch.arange(INTER * HIDDEN, dtype=torch.float32).reshape(INTER, HIDDEN) + 1000 * L + 100 * i
            down = torch.arange(HIDDEN * INTER, dtype=torch.float32).reshape(HIDDEN, INTER) - 1000 * L - 100 * i
            sd[f"{base}.{i}.up_proj.weight"] = up.to(dtype)  # nn.Linear(hidden -> inter): (out, in)
            sd[f"{base}.{i}.down_proj.weight"] = down.to(dtype)  # nn.Linear(inter -> hidden)
        sd[f"{prefix}.layers.{L}.mixer.gate.weight"] = torch.randn(E, HIDDEN).to(dtype)
    sd[f"{prefix}.embeddings.weight"] = torch.randn(32, HIDDEN).to(dtype)
    sd[f"{prefix}.layers.0.mixer.in_proj.weight"] = torch.randn(8, HIDDEN).to(dtype)
    return sd


def model_shapes(transposed=False):
    """What the live model expects: one grouped param per layer and projection."""
    up = (E, HIDDEN, INTER) if transposed else (E, INTER, HIDDEN)
    down = (E, INTER, HIDDEN) if transposed else (E, HIDDEN, INTER)
    shapes = {}
    for L in LAYERS:
        shapes[f"model.layers.{L}.mixer.experts.up_proj"] = up
        shapes[f"model.layers.{L}.mixer.experts.down_proj"] = down
        shapes[f"model.layers.{L}.mixer.gate.weight"] = (E, HIDDEN)
        # LoRA adapter params exist on the model but never in the base checkpoint
        shapes[f"model.layers.{L}.mixer.q_proj.lora_A.weight"] = (2, HIDDEN)
    shapes["model.embeddings.weight"] = (32, HIDDEN)
    shapes["model.layers.0.mixer.in_proj.weight"] = (8, HIDDEN)
    return shapes


def test_stacks_per_expert_tensors_in_order():
    sd = fake_checkpoint()
    original = {k: v.clone() for k, v in sd.items()}
    report = mp.merge_grouped_experts(sd, model_shapes())

    assert report.ok
    assert len(report.merged) == 2 * len(LAYERS)
    assert sum(p.num_experts for p in report.merged) == 2 * len(LAYERS) * E
    for L in LAYERS:
        base = f"model.layers.{L}.mixer.experts"
        up, down = sd[f"{base}.up_proj"], sd[f"{base}.down_proj"]
        assert up.shape == (E, INTER, HIDDEN) and down.shape == (E, HIDDEN, INTER)
        for i in range(E):
            assert torch.equal(up[i], original[f"{base}.{i}.up_proj.weight"])
            assert torch.equal(down[i], original[f"{base}.{i}.down_proj.weight"])
    # per-expert keys were popped; everything else untouched
    assert not [k for k in sd if ".experts." in k and k.endswith(".weight")]
    for k in ("model.embeddings.weight", "model.layers.0.mixer.in_proj.weight", "model.layers.1.mixer.gate.weight"):
        assert torch.equal(sd[k], original[k])
    # after the merge every base param of the model is present; only LoRA params are missing
    missing = set(model_shapes()) - set(sd)
    assert missing and all(".lora_" in k for k in missing)


def test_without_merge_the_experts_would_be_silently_missing():
    """The bug: intersecting checkpoint keys with model params drops every expert."""
    sd = fake_checkpoint()
    params = set(model_shapes())
    before = params & set(sd)
    mp.merge_grouped_experts(sd, model_shapes())
    after = params & set(sd)
    assert len(after) - len(before) == 2 * len(LAYERS)


def test_transposed_grouped_layout():
    sd = fake_checkpoint()
    original = {k: v.clone() for k, v in sd.items()}
    report = mp.merge_grouped_experts(sd, model_shapes(transposed=True))
    assert report.ok and all(p.transpose for p in report.merged)
    up = sd["model.layers.3.mixer.experts.up_proj"]
    assert up.shape == (E, HIDDEN, INTER)
    for i in range(E):
        assert torch.equal(up[i], original[f"model.layers.3.mixer.experts.{i}.up_proj.weight"].t())
    assert up.is_contiguous()


def test_shape_mismatch_is_reported_and_not_merged():
    sd = fake_checkpoint()
    shapes = model_shapes()
    shapes["model.layers.1.mixer.experts.up_proj"] = (E, INTER + 1, HIDDEN)
    report = mp.merge_grouped_experts(sd, shapes)
    assert not report.ok
    assert [n for n, _ in report.mismatched] == ["model.layers.1.mixer.experts.up_proj"]
    assert "model.layers.1.mixer.experts.up_proj" not in sd
    assert "model.layers.1.mixer.experts.0.up_proj.weight" in sd  # left alone


def test_inconsistent_expert_shapes_are_reported():
    sd = fake_checkpoint()
    sd["model.layers.1.mixer.experts.2.down_proj.weight"] = torch.zeros(HIDDEN, INTER + 2)
    report = mp.merge_grouped_experts(sd, model_shapes())
    assert [n for n, _ in report.mismatched] == ["model.layers.1.mixer.experts.down_proj"]


def test_missing_expert_skips_param():
    sd = fake_checkpoint()
    del sd["model.layers.3.mixer.experts.2.up_proj.weight"]
    report = mp.merge_grouped_experts(sd, model_shapes())
    assert report.ok
    assert [n for n, _ in report.skipped] == ["model.layers.3.mixer.experts.up_proj"]
    assert "model.layers.3.mixer.experts.up_proj" not in sd
    assert "model.layers.3.mixer.experts.0.up_proj.weight" in sd


def test_noop_when_grouped_param_already_present():
    sd = fake_checkpoint()
    mp.merge_grouped_experts(sd, model_shapes())
    snapshot = dict(sd)
    report = mp.merge_grouped_experts(sd, model_shapes())
    assert report.merged == [] and report.skipped == [] and report.ok
    assert sd.keys() == snapshot.keys()


def test_dry_run_changes_nothing():
    sd = fake_checkpoint()
    keys = set(sd)
    report = mp.merge_grouped_experts(sd, model_shapes(), dry_run=True)
    assert len(report.merged) == 2 * len(LAYERS)
    assert set(sd) == keys


def test_bf16_dtype_preserved():
    sd = fake_checkpoint(dtype=torch.bfloat16)
    mp.merge_grouped_experts(sd, model_shapes())
    assert sd["model.layers.1.mixer.experts.up_proj"].dtype == torch.bfloat16


def test_target_shapes_from_model_strips_wrapper_prefix():
    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = torch.nn.Parameter(torch.empty(E, INTER, HIDDEN))

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._checkpoint_wrapped_module = torch.nn.Module()
            self._checkpoint_wrapped_module.experts = Experts()

    shapes = mp.target_shapes_from_model([Wrapped()])
    assert shapes == {"experts.up_proj": (E, INTER, HIDDEN)}


def test_infer_target_shapes():
    shapes = {k: tuple(v.shape) for k, v in fake_checkpoint().items()}
    t = mp.infer_target_shapes(shapes)
    assert t == {k: v for k, v in model_shapes().items() if k.endswith(("up_proj", "down_proj"))}
    t2 = mp.infer_target_shapes(shapes, layout="transposed")
    assert t2["model.layers.1.mixer.experts.up_proj"] == (E, HIDDEN, INTER)
    with pytest.raises(ValueError, match="expected 8"):
        mp.infer_target_shapes(shapes, num_experts=8)
    del shapes["model.layers.1.mixer.experts.0.up_proj.weight"]
    with pytest.raises(ValueError, match="not contiguous"):
        mp.infer_target_shapes(shapes)


# ----------------------------------------------------------------- CLI end-to-end

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:  # core tests above still run without safetensors
    safe_open = save_file = None

needs_safetensors = pytest.mark.skipif(save_file is None, reason="safetensors not installed")


def write_sharded(sd, path: Path, n_shards=2):
    """Write sd as n shards, deliberately splitting a layer's experts across shards."""
    path.mkdir(parents=True)
    keys = list(sd)
    chunk = (len(keys) + n_shards - 1) // n_shards
    weight_map = {}
    for s in range(n_shards):
        name = f"model-{s + 1:05d}-of-{n_shards:05d}.safetensors"
        part = {k: sd[k].contiguous() for k in keys[s * chunk : (s + 1) * chunk]}
        save_file(part, str(path / name), metadata={"format": "pt"})
        weight_map.update({k: name for k in part})
    (path / mp.INDEX_FILENAME).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    (path / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))


def read_all(path: Path):
    out = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                out[k] = h.get_tensor(k)
    return out


@needs_safetensors
def test_cli_dry_run_writes_nothing(tmp_path, capsys):
    src = tmp_path / "ckpt"
    write_sharded(fake_checkpoint(), src)
    out = tmp_path / "out"
    assert mp.main([str(src), str(out), "--dry-run"]) == 0
    assert not out.exists()
    text = capsys.readouterr().out
    assert "MERGE model.layers.1.mixer.experts.up_proj (4, 10, 6) <- 4 x (10, 6)" in text
    assert "merged 4 grouped params from 16 per-expert tensors" in text


@needs_safetensors
def test_cli_roundtrip_with_rename(tmp_path):
    sd = fake_checkpoint(prefix="backbone", dtype=torch.bfloat16)
    src, out = tmp_path / "ckpt", tmp_path / "out"
    write_sharded(sd, src, n_shards=3)
    rc = mp.main([str(src), str(out), "--rename-prefix", "backbone.=model.", "--num-experts", str(E), "--max-shard-size", "1KB"])
    assert rc == 0

    merged = read_all(out)
    expected = {k: v for k, v in model_shapes().items() if ".lora_" not in k}
    assert {k: tuple(v.shape) for k, v in merged.items()} == expected
    for L in LAYERS:
        for i in range(E):
            assert torch.equal(
                merged[f"model.layers.{L}.mixer.experts.down_proj"][i],
                sd[f"backbone.layers.{L}.mixer.experts.{i}.down_proj.weight"],
            )
    assert merged["model.layers.1.mixer.experts.up_proj"].dtype == torch.bfloat16
    index = json.loads((out / mp.INDEX_FILENAME).read_text())
    assert set(index["weight_map"]) == set(merged)
    assert all((out / f).exists() for f in set(index["weight_map"].values()))
    assert len(set(index["weight_map"].values())) > 1  # small shard size forced several shards
    assert json.loads((out / "config.json").read_text()) == {"model_type": "nemotron_h"}


@needs_safetensors
def test_cli_with_target_shapes_json_and_mismatch(tmp_path):
    src = tmp_path / "ckpt"
    write_sharded(fake_checkpoint(), src)
    shapes = model_shapes(transposed=True)
    good = tmp_path / "shapes.json"
    good.write_text(json.dumps(shapes))
    assert mp.main([str(src), str(tmp_path / "out"), "--target-shapes", str(good)]) == 0
    merged = read_all(tmp_path / "out")
    assert merged["model.layers.3.mixer.experts.up_proj"].shape == (E, HIDDEN, INTER)

    shapes["model.layers.3.mixer.experts.up_proj"] = (E, 7, 7)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(shapes))
    assert mp.main([str(src), str(tmp_path / "out2"), "--target-shapes", str(bad)]) == 2
    assert not (tmp_path / "out2").exists()


@needs_safetensors
def test_cli_refuses_nonempty_output(tmp_path):
    src, out = tmp_path / "ckpt", tmp_path / "out"
    write_sharded(fake_checkpoint(), src)
    out.mkdir()
    (out / "keep.txt").write_text("x")
    assert mp.main([str(src), str(out)]) == 1
    assert mp.main([str(src), str(src)]) == 1
