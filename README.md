# Nemotron-H MoE LoRA: expert weights silently not loaded by NeMo Automodel

I fine-tuned NVIDIA's Nemotron 3.5 Lightning 30B-A3B (a hybrid Mamba/attention MoE, `model_type: nemotron_h`) with LoRA in NeMo RL. The run started cleanly, but the model it was training had every routed expert at random init. This repo is my write-up of how I found that, plus the fix I used (`mergepatch.py`, with tests).

## TL;DR

- **Symptom:** validation loss at step 0 was 13.17, which is no better than guessing uniformly over the vocabulary. Nothing errored or warned.
- **Cause:** transformers 5.x stores NemotronH's routed experts as one grouped tensor per layer (`...mixer.experts.up_proj`, shape `(n_experts, out, in)`). The checkpoint stores one tensor per expert (`...mixer.experts.{i}.up_proj.weight`). NeMo Automodel only merges these for model types listed in `MODELS_REQUIRING_TENSOR_MERGING`, and `nemotron_h` is not on that list. The loader runs with `strict=False`, so all 46 grouped expert parameters (every routed expert in every MoE layer) were dropped silently. Only 332 of the model's 426 parameters matched a checkpoint key; the other 48 misses were the new LoRA adapters, which is expected.
- **Fix:** before the state dict is loaded, stack the per-expert tensors into the grouped layout. Each stack is shape-checked against the live parameter, and each source tensor is popped as it is stacked so memory stays flat.
- **Result:** baseline val loss dropped from 13.17 to 1.6567, the stock model's real loss. Two epochs of LoRA then reached 0.7756, and a rerun with the same seed and config reached 0.7751.

## Environment

| | |
|---|---|
| Model | Nemotron 3.5 Lightning 30B-A3B, BF16 HF checkpoint (~62 GB), `model_type: nemotron_h` |
| Framework | NeMo RL container `nvcr.io/nvidia/nemo-rl:v0.7.0` (`nemo_rl 0.6.0+19244a094`), SFT with LoRA |
| Backend | DTensor / FSDP2 ("automodel" path), `automodel_kwargs.force_hf: true` |
| Bundled Automodel | `NVIDIA-NeMo/Automodel@24b47e8` (2026-04-30) |
| Libraries | transformers 5.8.1, torch 2.11.0+cu130, NCCL 2.28.9 |
| Hardware | 2 × NVIDIA GB10 (DGX Spark), ~121 GB usable unified memory each, 200 Gb/s link between them; Ray cluster across both nodes, FSDP sharding (~31 GB of weights per node) |
| Data | a private domain corpus in OpenAI chat format (not included here) |
| Dates | 2026-08-15 to 2026-08-17 |

## Symptom

The 2-node run came up without errors: NCCL rendezvous, sharded load, a validation pass and a real training step on both GPUs. But:

- val loss at step 0 was **13.17**. A model that guesses uniformly over the vocabulary scores ln(vocab_size), and a randomly initialised network lands around there or a little above. This one was in that range.
- I noticed the checkpoint used `backbone.*` keys while the model used `model.*` keys, so I re-keyed it. Val loss moved to **12.57**: still effectively random.

So the weights were not reaching the sharded parameters. I aborted the run rather than spend GPU-hours training on top of noise.

## How I isolated it

1. **Ruled out the checkpoint.** I loaded the re-keyed checkpoint with plain transformers, outside NeMo, using `output_loading_info=True`. It reported **0 missing and 0 unexpected keys**, and a greedy completion of a capital-of-France prompt returned "Paris is the capital of France." So the checkpoint and the HF model class were both fine, and the fault had to be in the training loader.
2. **Dropped a wrong hypothesis.** I first suspected HF's `_init_weights` was re-initialising weights after the load (the `_is_hf_initialized` flag). It wasn't.
3. **Instrumented the loader.** In Automodel's `_load_full_state_dict_into_model` I printed the intersection of model parameter names and state-dict keys just before `set_model_state_dict`. It showed **332 / 426**. The 94 misses broke down as:
   - 46 grouped expert params (`up_proj` and `down_proj` for each MoE layer), which never matched because the checkpoint names them per expert;
   - 48 LoRA adapter params, which are expected to be missing because they're new.

The load uses `StateDictOptions(strict=False, full_state_dict=True)`. Unmatched keys on both sides were dropped with no log line.

## Root cause

transformers' native `NemotronHForCausalLM` holds the routed experts in `NemotronHExperts`, which has two parameters per layer:

```
model.layers.N.mixer.experts.up_proj     (n_experts, intermediate, hidden)
model.layers.N.mixer.experts.down_proj   (n_experts, hidden, intermediate)
```

The HF checkpoint has one `nn.Linear` weight per expert:

```
backbone.layers.N.mixer.experts.{i}.up_proj.weight     (intermediate, hidden)
backbone.layers.N.mixer.experts.{i}.down_proj.weight   (hidden, intermediate)
```

transformers bridges the two in its own `from_pretrained`. Its `conversion_mapping.py` has a `nemotron_h` entry with `WeightRenaming("backbone.", "model.")` and `MergeModulelist(dim=0)` converters, which is why the plain-transformers load in step 1 was clean.

NeMo Automodel only routes a checkpoint through those converters (`_convert_checkpoint_with_transformers`) when `requires_tensor_merging(model_type)` is true, which means the model type is in `MODELS_REQUIRING_TENSOR_MERGING` (in `nemo_automodel/components/checkpoint/conversion_mapping.py`). That set covers Mixtral, Qwen-MoE, DeepSeek, Jamba and others, but not `nemotron_h`. My NemotronH load therefore went down the generic full-state-dict path. That path ran neither the rename nor the expert merge, and `strict=False` hid the result. After the re-key, attention, Mamba, router, embedding and norm weights loaded correctly. Every routed-expert FFN stayed random, and the network's output was no better than random guessing.

In the bundled Automodel (`24b47e8`), the relevant lines are: the merge set at `conversion_mapping.py` L60–81, the merge branch at `checkpointing.py` L421, the full-state-dict path at L445–484, and `strict=False` in `_load_full_state_dict_into_model` at L1430.

## The fix

[`mergepatch.py`](mergepatch.py) applies this rule to every model parameter named `<prefix>.experts.<proj>` (proj ∈ gate_proj / up_proj / down_proj) that is not already in the state dict:

1. Take `n_experts` from the parameter's leading dimension. For an FSDP2 DTensor, `.shape` is the full logical shape, so this works on sharded models.
2. Collect `<prefix>.experts.{i}.<proj>.weight` for `i in range(n_experts)`. If any of them is missing, skip the parameter.
3. Shape-check. If the parameter's trailing dims equal the per-expert `(out, in)` shape, stack the experts as they are. If they equal the transpose, transpose each expert, then stack. If neither matches, report the parameter and don't merge it.
4. Pop each per-expert tensor as it goes into the stack, then store `torch.stack(..., dim=0)` under the grouped name.

During the run, I injected this loop into the container's copy of `_load_full_state_dict_into_model`, right before `set_model_state_dict`. The log reported 46 grouped params merged and the intersection rose to **378 / 426**. The 48 that were still missing were the LoRA adapters, which are supposed to start fresh.

The file here is a cleaned-up, standalone version of that patch. The merge logic is unchanged: the same name pattern, the same shape check with a transpose fallback, and the same pop-as-you-stack. Since then I've added two things:

- a library API you can call from a loader:

  ```python
  from mergepatch import merge_grouped_experts, target_shapes_from_model

  report = merge_grouped_experts(state_dict, target_shapes_from_model(model_parts))
  print(report.summary())
  if not report.ok:
      raise ValueError(report.mismatched)   # the original printed and skipped
  set_model_state_dict(part, model_state_dict=state_dict, options=options)
  ```

- an offline CLI that rewrites a safetensors checkpoint directory. It can also re-key prefixes.

  ```bash
  # Show what would be merged. Only safetensors headers are read; nothing is written.
  python mergepatch.py /path/to/hf-checkpoint --dry-run

  # Re-key backbone.* -> model.* and write a grouped-expert copy (input is left untouched)
  python mergepatch.py /path/to/hf-checkpoint /path/to/out \
      --rename-prefix backbone.=model. --num-experts N

  # Shape-check against shapes dumped from the live model, as the in-loader version did
  python mergepatch.py /path/to/hf-checkpoint /path/to/out --target-shapes shapes.json
  ```

  The CLI exits with code 2 and writes nothing if any shape doesn't reconcile. Without `--target-shapes`, it infers the grouped layout as `(n_experts, out, in)`, the layout used by transformers' `NemotronHExperts`. Pass `--layout transposed` for `(n_experts, in, out)`.

Caveat: the run itself used only the in-process form. The offline CLI is new, and so far I've tested it on synthetic tensors only, not on the 62 GB checkpoint. Keep your original checkpoint.

### Tests

```bash
python -m pip install torch safetensors pytest
python -m pytest -q
```

The tests use tiny CPU tensors and need no GPU or model download. They check:

- per-expert tensors land in the right slot, in the right order;
- the transposed layout works;
- shape mismatches, inconsistent experts and missing experts are reported and not merged;
- dry runs, re-running, and bf16 are all safe;
- the CLI round-trips a sharded checkpoint with re-keying.

They pass on Python 3.12, torch 2.14.1 and safetensors 0.8.0 (15 passed).

## Results

| | val loss |
|---|---|
| Step 0, before the fix | 13.17 (12.57 after re-key) |
| Step 0, after the fix (stock model) | **1.6567** |
| After 2 epochs of LoRA, run 1 | 0.7756 |
| After 2 epochs, rerun with the same seed and config | **0.7751** |

- Training: about 287 s/step at global batch 16 and sequence length 2048, over NCCL sockets. Two epochs took about 12 h.
- Run 1's weights were lost because of my own setup mistake. The consolidated save happens on rank 0, which landed on the node whose container had no results mount, and I tore the containers down before checking. The rerun reproduced the loss to within 0.0005.
- Adapter: 3.6 MB, PEFT-format LoRA, 24 modules (attention projections only), r=16, alpha=32, bf16.
- Serving: the adapter runs as a vLLM LoRA module (`--enable-lora --max-lora-rank 16 --lora-modules <name>=<path>`). The stock model is served alongside it unchanged.
- Quality: on held-out prompts, the adapter's outputs moved in the trained direction with no visible loss of fluency. This was a qualitative check only; I have not published an eval.

## Other problems I hit on the way

Each of these has a patch or workaround that worked for me in that container. I haven't checked each one against current upstream.

**Two-node setup**
1. NCCL's bundled InfiniBand net plugin crashed on GB10 (`ncclNetInit` / `ncclIbFinalize`). Workaround: `NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_SOCKET_IFNAME=<iface>`. I haven't tuned RoCE yet.
2. The Automodel parallelizer's NemotronH plan referenced `model.backbone.layers`, but the native transformers class has `model.model.layers`, which raised an `AttributeError`. I patched the path.
3. `automodel_kwargs.force_hf: true` was required, because the custom NemotronH class also broke the parallel plan in that build.
4. transformers' `initialization.py` `copy_` is not DTensor-aware, so `_init_weights` under FSDP2 failed with "mixed torch.Tensor and DTensor". I patched it to handle DTensor.
5. LoRA `target_modules` needed full-path wildcards (`"*.q_proj"`); otherwise the optimizer received an empty parameter list. With a bf16 base, the LoRA dtype was hard-coded to fp32, and fp32 leaked through the Mamba path, causing matmul dtype errors. I patched `lora_dtype` to bf16 and added an input cast at `lm_head`.
6. Checkpoint keys were `backbone.*` but the model expected `model.*`. I made a re-keyed copy (`--rename-prefix backbone.=model.` does the same). transformers' own `nemotron_h` mapping contains this rename, but for the reason above it never ran.

**Single node (why I went to two)**
- The automodel `setup.py` loads the base model in fp32 for master weights. That's about 124 GB on a box with 121 GB usable, and `automodel_kwargs` can't override it (duplicate kwarg). For LoRA-only training I patched it to bf16.
- Ray's memory monitor reads cgroup memory, which counts page cache. Streaming 62 GB of safetensors hit its kill threshold in seconds, while real process memory was about 3 GB. Workaround: `RAY_memory_monitor_refresh_ms=0`.
- Even in bf16 with `low_cpu_mem_usage`, FSDP2/DTensor materialization exhausted the node. NemotronH had no gradient-checkpointing support in that Automodel build, and CUDA managed memory doesn't show up in RSS or `free`, so the growth was hard to see. Sharding across two nodes solved it. A separate report of the single-GB10 load problem exists upstream: NVIDIA-NeMo/Automodel#3532 (not mine, closed).

**Config gotchas**
- The NeMo RL SFT config is an overlay. Its first line must be `defaults: /opt/nemo-rl/examples/configs/sft.yaml` (the path inside the container); otherwise validation fails with missing-field errors.
- A top-level `data.train_data_path` is silently ignored, and the run **trains on SQuAD** instead. The correct form is `data.train: {dataset_name: openai_format, data_path: ..., chat_key: messages}`, and the same for validation.
- `policy.tokenizer.chat_template: "default"` (a string) selects the model's own template; `null` fails validation.
- There is no `latest` tag for `nemo-rl` on NGC, so pin a version.

## Status upstream (checked 2026-10-14)

- **Not fixed on `main`.** At `NVIDIA-NeMo/Automodel@03a8b7e`, `nemotron_h` is still missing from `MODELS_REQUIRING_TENSOR_MERGING` (`conversion_mapping.py` L62–83). `requires_tensor_merging` is at L86–100, and it's called from `checkpointing.py` L1081, L1676 and L2147. `_load_full_state_dict_into_model` still loads with `strict=False` (L2840; load calls at L2853 and L2859) and doesn't report misses.
- **transformers already has the converter.** transformers v5.8.1 and current `main` both define the `nemotron_h` `MergeModulelist(dim=0)` converters, so the Automodel fix is mostly wiring.
- **The loader has changed since the build I used.** On `main`, standard HF safetensors models normally go through a DCP path. That path logs `Checkpoint key mismatch ... missing=N` and loads strictly for single-part models, so on `main` this may now fail loudly instead of silently. I have not re-run on `main`.
- **The custom path isn't affected.** Automodel's custom Nemotron V3 implementation (`force_hf: false`, `NemotronV3StateDictAdapter`) merges experts itself. The gap is the `force_hf: true` / native-transformers path.
- **No existing report found.** I searched NVIDIA-NeMo/Automodel and NVIDIA-NeMo/RL issues and PRs for `nemotron_h`, `requires_tensor_merging`, `MODELS_REQUIRING_TENSOR_MERGING`, "experts not loaded" and similar terms. The closest is #3532, which is about memory, not this.
- **Reported upstream:** [NVIDIA-NeMo/Automodel#4357](https://github.com/NVIDIA-NeMo/Automodel/issues/4357), with a suggested patch against `main`: add `nemotron_h` to the set, with a guard for the remote-code class, and warn on unloaded base params.

## Repo contents

| File | What |
|---|---|
| `mergepatch.py` | The merge (library + CLI, `--dry-run`) |
| `tests/test_mergepatch.py` | CPU-only tests with tiny fake tensors |

MIT licensed. Copyright (c) 2026 Joseph Crabtree.
