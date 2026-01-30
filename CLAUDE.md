# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Megatron Bridge is a **distributed training framework** that uses HuggingFace as its checkpoint interchange format and Megatron Core as its distributed execution backend. The "bridge" is the entry/exit point—conversion utilities that let you start from any HuggingFace checkpoint, train at scale with Megatron Core's parallelism primitives, and export back to HuggingFace format.

**Core value proposition:**
- **Recipes**: Production-ready training configs encoding NVIDIA's optimization expertise
- **Training orchestration**: Pretraining, SFT, LoRA/DoRA, knowledge distillation
- **Parallelism abstraction**: TP/PP/VP/CP/EP without manual sharding code
- **Fault tolerance**: In-process restart, async checkpointing, straggler detection
- **Mixed precision**: FP8, MXFP8, FP4, BF16 with adaptive scaling

## Supported Models

**Text Models**: Llama (2/3/3.1/3.2), Gemma (1/2/3), Qwen (1.5/2/2.5/3), Mistral, Ministral, DeepSeek, GLM, Kimi, Nemotron, Llama-Nemotron, OLMoE

**Vision-Language Models**: Qwen-VL, Nemotron-VL, Gemma3-VL, GLM-VL

## When to Use Megatron Bridge

**Use Megatron Bridge when:**
- Training at multi-node scale where HuggingFace Trainer becomes a bottleneck
- You need TP/PP/CP parallelism (HuggingFace only does FSDP/DP)
- Starting from HuggingFace checkpoints but need Megatron Core performance
- You want NVIDIA's production-tested recipes rather than tuning from scratch

**Consider alternatives when:**
- Single-GPU or small-scale training (HuggingFace Trainer is simpler)
- You're already deep in Megatron-LM ecosystem (use that directly)
- You need models not yet bridged (check `src/megatron/bridge/models/`)

## Common Commands

### Development Setup

```bash
# Build Docker container for development
docker build -f docker/Dockerfile.ci -t megatron-bridge .

# Run container with GPU access
docker run --rm -it -w /workdir -v $(pwd):/opt/Megatron-Bridge \
  --entrypoint bash --gpus all megatron-bridge

# Install pre-commit hooks (requires uv)
uv run --group dev pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run only unit tests
uv run pytest tests/unit_tests/

# Run only functional tests
uv run pytest tests/functional_tests/

# Run specific test markers
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m "not system"

# Run tests with timeout (useful for long-running tests)
uv run pytest --timeout=300 tests/functional_tests/
```

### Linting and Formatting

```bash
# Run ruff to check and auto-fix linting issues
uv run ruff check --fix .

# Format code with ruff
uv run ruff format .

# Run both linting and formatting
uv run ruff check --fix . && uv run ruff format .
```

### Training

```bash
# Single-node pretraining with recipe
torchrun --nproc-per-node=8 scripts/training/run_recipe.py \
  --recipe llama32_1b_pretrain_config

# Finetuning with CLI overrides
torchrun --nproc-per-node=8 scripts/training/run_recipe.py \
  --recipe llama32_1b_finetune_config \
  train.train_iters=5000 \
  optimizer.lr=0.0002 \
  model.tensor_model_parallel_size=2

# Multi-node training with NeMo-Run (local test)
python scripts/training/launch_with_nemo_run.py \
  --local \
  --script run_recipe.py \
  --recipe llama32_1b_pretrain_config \
  --devices 2 \
  train.train_iters=10

# Multi-node training on Slurm
python scripts/training/launch_with_nemo_run.py \
  --script run_recipe.py \
  --recipe llama3_8b_pretrain_config \
  --nodes 4 \
  --devices 8 \
  --partition gpu \
  --account my_account
```

### Checkpoint Conversion

```bash
# Convert HuggingFace to Megatron (see examples/conversion/)
uv run python examples/conversion/convert_checkpoints.py \
  --hf-model meta-llama/Llama-3.2-1B \
  --output-dir ./megatron_checkpoints

# Round-trip conversion test (HF -> Megatron -> HF)
uv run python examples/conversion/hf_megatron_roundtrip.py \
  --model-name meta-llama/Llama-3.2-1B

# Multi-GPU conversion with parallelism
torchrun --nproc-per-node=4 examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
  --model-name meta-llama/Llama-3-8B \
  --tp 4 --pp 1

# Compare HuggingFace and Megatron model outputs
uv run python examples/conversion/compare_models.py \
  --hf-model meta-llama/Llama-3.2-1B
```

### Running a Single Test

```bash
# Run a specific test file
uv run pytest tests/unit_tests/models/test_model_bridge.py

# Run a specific test function
uv run pytest tests/unit_tests/models/test_model_bridge.py::test_llama_bridge

# Run with verbose output
uv run pytest -v tests/unit_tests/models/test_model_bridge.py

# Run with print statements visible
uv run pytest -s tests/unit_tests/models/test_model_bridge.py
```

## High-Level Architecture

### 1. The Bridge Concept

Megatron Bridge provides **bidirectional conversion** between HuggingFace and Megatron Core formats. The architecture consists of:

- **AutoBridge**: High-level API that auto-detects model architecture and selects the appropriate bridge
- **MegatronModelBridge**: Base class for model-specific bridges (e.g., LlamaBridge, QwenBridge)
- **MegatronMappingRegistry**: Parameter name mapping system that handles wildcard patterns and nested modules
- **MegatronParamMapping**: Weight transformation strategies (direct copy, TP scatter/gather, QKV fusion, etc.)

**Key Pattern**: Bridges use a **registration-based dispatch** system where each bridge registers itself for specific (HuggingFace, Megatron) model pairs. This allows AutoBridge to automatically select the correct bridge without manual routing logic.

```python
@MegatronModelBridge.register_bridge(source=LlamaForCausalLM, target=GPTModel)
class LlamaBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):
        # Convert HF config to Megatron provider
        return LlamaModelProvider(...)

    def mapping_registry(self):
        # Define parameter mappings
        return MegatronMappingRegistry(...)
```

### 2. Providers: Configuration + Instantiation

**Providers** are dataclass-based configuration objects that know how to instantiate distributed Megatron models. They serve three roles:

1. **Configuration holder**: Store model architecture parameters (hidden_size, num_layers, etc.)
2. **Parallelism manager**: Hold TP/PP/VP/CP sizes (set lazily before instantiation)
3. **Model factory**: `provide()` method creates actual Megatron Core models

**Key insight**: Parallelism is configured *after* provider creation but *before* model instantiation, allowing dynamic scaling without re-creating the bridge.

```python
# Create provider from HuggingFace model
bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")
provider = bridge.to_megatron_provider()

# Configure parallelism (lazy - not baked in yet)
provider.tensor_model_parallel_size = 4
provider.pipeline_model_parallel_size = 2

# Instantiate distributed model
model = provider.provide_distributed_model(wrap_with_ddp=True)
```

### 3. Training Loop Architecture

The training loop is a **configuration-driven composition** with clear separation of concerns:

- **setup()**: Initialize distributed backend, create model/optimizer/scheduler, load checkpoints
- **train()**: Main loop with forward/backward/optimizer steps, checkpointing, and validation
- **forward_step()**: User-provided function for loss computation (supports multiple signatures)

**ConfigContainer** is the central configuration object containing nested configs for training, optimizer, data, checkpointing, parallelism, etc. Each subconfig supports **deferred post-init** allowing modifications before finalization.

**Callback system**: The training loop supports callbacks for extensibility (custom logging, early stopping, profiling).

### 4. Recipes: Factory Functions for Reproducibility

Recipes are **factory functions** that return pre-configured ConfigContainer objects:

```python
def llama3_8b_pretrain_config(**user_kwargs) -> ConfigContainer:
    """Pre-configured recipe for Llama 3 8B pretraining."""
    recommended_kwargs = {
        "hf_path": "meta-llama/Meta-Llama-3-8B",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        # ... more defaults
    }
    combined_kwargs = {**recommended_kwargs, **user_kwargs}
    return _llama3_common(**combined_kwargs)
```

**Hierarchical organization**:
- Base common function with all parameters
- Size-specific variants (8B, 70B, 405B)
- Sequence-length variants (16K, 64K, 128K)
- Precision variants (FP8, MXFP8, NVFP4)

Recipes encode NVIDIA's recommended settings while allowing full user customization via TypedDict parameters.

### 5. Parallelism Dimensions

Megatron Bridge supports multiple parallelism strategies that can be combined:

- **Tensor Parallelism (TP)**: Splits each layer horizontally across GPUs (e.g., attention heads, FFN columns)
- **Pipeline Parallelism (PP)**: Splits model layers vertically across GPUs
- **Virtual Pipeline Parallelism (VP)**: Multiple model chunks per GPU for better pipeline utilization
- **Context Parallelism (CP)**: Splits sequence length across GPUs (for long contexts)
- **Data Parallelism (DP)**: Implicit via DDP - replicates model across remaining GPUs

**Models are PP-stage-aware**: Each pipeline stage gets a specialized model with correct embedding/output layers. The `provide_distributed_model()` method returns a list of models (one per VP stage).

### 6. Weight Conversion Workflow

**HuggingFace → Megatron**:
1. Iterate through Megatron model parameters
2. Query MappingRegistry for HF source names (supports wildcards)
3. Load HF weights and apply transformations (transpose, reshape, QKV fusion, gated MLP merge)
4. Scatter across TP ranks if needed
5. Assign to Megatron parameter

**Megatron → HuggingFace**:
1. Iterate through HF parameters
2. Query MappingRegistry for Megatron sources
3. Gather shards from all TP ranks
4. Broadcast from owning PP rank
5. Apply inverse transformations
6. Yield HF-format tensors (can save or stream)

**Transformation strategies**: DirectMapping, ColumnParallelMapping, RowParallelMapping, QKVMapping, GatedMLPMapping, AutoMapping (auto-detects based on module type).

### 7. Data Infrastructure

The data pipeline supports multiple sources with distributed loading:

- **HuggingFace datasets**: Load directly from HF Hub or local paths via dataset builders
- **Energon**: Integration with NVIDIA's Energon library for large-scale data
- **Data blending**: Combine multiple datasets with configurable ratios
- **Packed sequences**: Efficient packing for variable-length inputs
- **VLM datasets**: Specialized loaders for vision-language model training

### 8. Fault Tolerance

For long-running distributed training, Megatron Bridge integrates with `nvidia-resiliency-ext`:

- **In-process restart**: Recover from transient failures without restarting the job
- **Async checkpointing**: Non-blocking saves to minimize training interruption
- **Straggler detection**: Identify slow ranks that degrade overall throughput

This is critical for multi-day training runs where hardware failures are expected.

## Key Code Locations

### Bridge Infrastructure
- `src/megatron/bridge/models/conversion/auto_bridge.py` - AutoBridge API
- `src/megatron/bridge/models/conversion/model_bridge.py` - Base bridge class
- `src/megatron/bridge/models/conversion/param_mapping.py` - Weight transformation strategies
- `src/megatron/bridge/models/<model>/` - Model-specific bridges (llama, qwen, gemma, etc.)

### Training Framework
- `src/megatron/bridge/training/pretrain.py` - Main training loop
- `src/megatron/bridge/training/finetune.py` - Finetuning loop
- `src/megatron/bridge/training/config.py` - ConfigContainer and subconfigs
- `src/megatron/bridge/training/gpt_step.py` - GPT forward step
- `src/megatron/bridge/training/vlm_step.py` - Vision-language model forward step

### Recipes
- `src/megatron/bridge/recipes/<model>/` - Model-specific recipes
- `src/megatron/bridge/recipes/utils/` - Shared recipe utilities (optimizer, dataset, checkpoint)

### Providers
- `src/megatron/bridge/models/model_provider.py` - Base provider pattern
- `src/megatron/bridge/models/gpt_provider.py` - GPT model provider
- `src/megatron/bridge/models/<model>/model_provider.py` - Model-specific providers

### Training Scripts
- `scripts/training/run_recipe.py` - Generic recipe launcher
- `scripts/training/launch_with_nemo_run.py` - NeMo-Run launcher (local/Slurm)
- `scripts/training/launch_with_sbatch.sh` - Direct sbatch launcher

### Examples
- `examples/conversion/` - Checkpoint conversion examples
- `examples/models/` - Model-specific training examples
- `tutorials/recipes/llama/` - Step-by-step Llama tutorials

### Data Infrastructure
- `src/megatron/bridge/data/loaders.py` - Distributed data loading with blend management
- `src/megatron/bridge/data/samplers.py` - Distributed samplers for training
- `src/megatron/bridge/data/builders/` - HuggingFace and finetuning dataset builders
- `src/megatron/bridge/data/finetuning.py` - SFT data pipeline with batch preparation
- `src/megatron/bridge/data/energon/` - Energon data library integration
- `src/megatron/bridge/data/vlm_datasets/` - Vision-language model datasets

### PEFT (Parameter-Efficient Fine-Tuning)
- `src/megatron/bridge/peft/lora.py` - LoRA implementation integrated with distributed training
- `src/megatron/bridge/peft/dora.py` - DoRA (Dimension-wise LoRA) variant
- `src/megatron/bridge/peft/base.py` - Abstract PEFT base class with model transformation pipeline
- `src/megatron/bridge/peft/recompute.py` - Gradient recomputation for memory efficiency

### Fault Tolerance and Resilience
- `src/megatron/bridge/training/fault_tolerance.py` - nvidia-resiliency-ext integration
- `src/megatron/bridge/training/inprocess_restart.py` - Graceful recovery without job restart
- `src/megatron/bridge/training/nvrx_straggler.py` - Straggler detection and mitigation
- `src/megatron/bridge/training/checkpointing.py` - Async checkpointing, fault-tolerant saves

### Inference
- `src/megatron/bridge/inference/vlm/` - VLM inference controllers and engines

## Important Implementation Patterns

### 1. Lazy Configuration
Parallelism parameters are not baked into providers at creation time. They can be modified after bridge creation but before model instantiation, enabling dynamic scaling.

### 2. Registration-Based Routing
Bridges auto-register themselves using decorators. AutoBridge uses type inspection to automatically select the correct bridge without explicit routing logic.

### 3. Typed Recipes with TypedDict
Recipes use TypedDict for parameter validation, providing IDE autocomplete and catching typos at development time.

### 4. Two-Phase Conversion
Checkpoint conversion uses a planning phase (build task list) and execution phase (load, transform, distribute), enabling memory-efficient streaming and parallelism-aware distribution.

### 5. Pipeline Specialization
Models created by providers are PP-stage-aware with correct embedding/output layers. Layer numbering is local to PP stage but requires global adjustment for checkpointing.

### 6. Callback System
Training loop supports callbacks with access to full training state (CallbackContext), enabling custom logging, monitoring, and dynamic adjustments.

### 7. Forward Step Flexibility
The training loop auto-detects forward step function signatures via type hints or parameter names, supporting 2-4 argument variants for different use cases.

## Code Style and Conventions

- **Python standard**: Python 3.10+ (use modern type hints: `X | None`, `dict[str, int]`)
- **Line length**: 119 characters max
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE_CASE for constants
- **String quotes**: Use double quotes (matching ruff formatter)
- **Imports**: Group in order: future, stdlib, third-party (torch, transformers, megatron.core), first-party (megatron.bridge), local
- **Docstrings**: Use Google style for classes and functions
- **Type hints**: Always use for function arguments and return types
- **uv commands**: Use `uv run` instead of activating virtualenv (except in docker/Dockerfile.ci)

### Commit Message Format

```
[{modules}] {type}: {description}

Modules: model, recipe, training, data, ckpt, peft, perf, ci, doc, test, build, misc
Types: feat, fix, refactor, chore, test

Example: [model] feat: Add Qwen3 model bridge
Example: [BREAKING][training] refactor: Change optimizer config structure
```

### Copyright Header

Add to all Python files:

```python
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## Testing Guidelines

- **Unit tests**: Place in `tests/unit_tests/`, aim to test functions in isolation without large artifacts
- **Functional tests**: Place in `tests/functional_tests/`, integration tests that may perform training or use HF checkpoints
- **Test markers**: Use `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.system` to categorize tests
- **Process isolation**: Use `subprocess.run` inside pytest functions when tests need process isolation

## Documentation Requirements

All new key features (new models, new parallelism strategies) must include documentation updates that:
- Explain motivation and purpose
- Outline technical approach and architecture
- Provide usage examples
- Document internal implementation details

Update `docs/index.md` when adding or renaming markdown files.

## Dependencies Management

Use `uv` for dependency management:

```bash
# Add required dependency
uv add <package>

# Add optional dependency to a specific extra group
uv add --optional --extra <extra> <package>

# Lock file is tracked in git
git add uv.lock pyproject.toml
git commit -m "build: Adding dependencies"
```

## CI and Pre-commit

- **Pre-commit**: Runs ruff linting/formatting, checks end-of-file, trailing whitespace
- **CI**: Automatically runs if commits are GPG-signed, otherwise comment `/ok to test <commit-SHA>` on PR
- **DCO**: All commits must be signed-off with `git commit -s` (Developer Certificate of Origin)

## Debugging Tips

**Checkpoint conversion issues:**
- Run `examples/conversion/compare_models.py` to verify numerical equivalence
- Check `MappingRegistry` for missing parameter mappings
- Use `--tp 1 --pp 1` first to isolate conversion from parallelism issues

**Distributed training hangs:**
- Set `NCCL_DEBUG=INFO` to see collective operations
- Check for mismatched tensor shapes across ranks
- Verify all ranks reach the same collective calls

**Memory issues:**
- Reduce `micro_batch_size` first
- Enable activation checkpointing via `recompute_granularity`
- Try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- For conversion: use streaming mode to avoid loading full checkpoint into memory

**Recipe debugging:**
- Print the `ConfigContainer` to see all resolved settings
- Override specific configs via CLI: `train.train_iters=10` for quick iteration
- Use `--local` with `launch_with_nemo_run.py` for single-node testing before scaling
