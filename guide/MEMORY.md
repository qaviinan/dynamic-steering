# Repo Memory

This file is a living map of the repository's purpose, flow, and historical pivots.
It is meant to preserve project context after multiple changes in direction, especially
where the checked-in code, filenames, and older docs no longer line up perfectly.

## Current Bottom Line

The repo started as an attempt to extract a harmful or refusal-related direction from
an already abliterated model and then use that direction to train probes for dynamic
steering at inference time.

That approach was abandoned because the abliterated model was not a reliable source of
the refusal signal. The project pivoted to extracting the refusal direction from the
baseline instruct model instead, then using inference-time interventions on that base
model.

Today, the most relevant end-to-end path is:

1. `run_refusal_extraction.py`
2. `data/steering_vectors.npz`
3. `train_probe.py`

The older abliterated-model pipeline still exists in the repo, but it is now better
treated as legacy or exploratory code rather than the main path.

## Project Evolution And Pivots

This section captures project memory from prior work and discussion, including context
that is not always obvious from the current files alone.

### Phase 1: The Abliteration Attempt

- Original goal: take an abliterated or uncensored model, extract the harmful or
  refusal-related direction from it, train probes on that direction, and dynamically
  steer the model during inference.
- Core assumption: even if a model had been abliterated, some geometric trace of the
  refusal feature might still be recoverable.
- What happened: that signal was effectively absent or unusable. Attempts to extract a
  direction from the abliterated model produced weak signal, zero-like behavior, or
  gibberish downstream.
- Practical conclusion: a successfully abliterated model is a poor source of truth for
  the refusal feature because the feature geometry may have already been removed.

### Phase 2: Infrastructure Buildout And Scaling Roadblocks

- The project then focused on the mechanics needed for steering:
  - loading models into `HookedTransformer`
  - formatting XSTest prompts
  - caching residual activations
  - computing difference-in-means vectors
  - training linear probes
- A major problem appeared when steering vectors were unit-normalized.
- Unit normalization removed the feature's natural magnitude, which created a bad
  operating regime:
  - small `alpha` values did almost nothing
  - large `alpha` values caused LayerNorm collapse and gibberish outputs
- Another issue was model-loading ambiguity:
  - older code paths could accidentally route back to the abliterated model through
    environment-variable overrides
  - newer refusal extraction avoids this by bypassing `setup_model.py` and loading the
    instruct model directly
- Important pivot:
  - difference-in-means vectors are now kept at native scale in
    `steering/steering_utils.py`
  - `alpha=1.0` is treated as a meaningful natural-magnitude intervention baseline
- Important infrastructure improvement:
  - activations are stored incrementally under SHA-derived filenames so extraction runs
    are resumable and do not need to replay old prompts

### Phase 3: Searching For The Best Layer

- The project moved away from assuming a fixed layer such as 14.
- Historical intent: evaluate layers and select the one where refusal is most
  separable, using metrics such as Cohen's d and ROC AUC.
- Current checked-in implementation in `run_refusal_extraction.py` caches all layers,
  then chooses the layer with the largest mean-difference norm before reporting
  diagnostics on the chosen layer.
- Practical takeaway:
  - historical project memory says "best layer by separability"
  - current code says "best layer by largest difference vector norm"
- This is one of the most important code-vs-history mismatches in the repo.

### Phase 4: Extreme Inference-Time Intervention

- The final major pivot was away from single-layer vector addition as the only
  intervention.
- The stronger intervention strategy is projection subtraction:
  - compute the refusal direction
  - normalize it to a unit vector for projection removal
  - subtract the component of each activation along that direction
- Intended high-level idea:
  - prevent the model from representing the refusal direction anywhere in the forward
    pass
- Current implementation in `train_probe.py`:
  - installs hooks on every layer's `hook_attn_out`
  - installs hooks on every layer's `hook_mlp_out`
  - projects the refusal direction out of those outputs across all layers and tokens
- Important nuance:
  - the historical description sometimes refers to every attention head output
  - the current code operates at the attention-module output and MLP output level, not
    per individual head tensor

### Current Interpretation Of The Project

- The project is no longer best described as "extract from the abliterated model and
  dynamically steer that same model."
- It is better described as:
  - extract refusal-related activations from the baseline instruct model
  - identify a useful direction and layer
  - test additive steering and stronger projection-removal interventions at inference
    time
- Probe-gated dynamic steering still exists conceptually, but the checked-in code is
  stronger as a research harness than as a productionized conditional guardrail.

## Main Execution Flow

The repo has two overlapping flows: a legacy abliterated-model flow and the newer
base-model refusal flow.

### Preferred Current Flow

1. Model loading
   - `run_refusal_extraction.py` loads the instruct model directly into
     `HookedTransformer`.
   - It intentionally bypasses `setup_model.py` so `MODEL_ID_ABL` cannot accidentally
     redirect the run to abliterated weights.

2. Dataset preparation
   - `data/format_xstest.py` loads XSTest from Hugging Face.
   - It writes `data/xstest_full.csv` with:
     - `focus`
     - `label`
     - `prompt`
     - `formatted_prompt`

3. Activation caching
   - `activations/cache_activations.py` runs `model.run_with_cache(...)` on the prompt
     only.
   - It extracts final-token `resid_post` activations.
   - `run_refusal_extraction.py` stores all layers for each prompt as
     `[num_layers, d_model]`.
   - Files are saved under `data/steering_state/{safe,unsafe}/{sha}.npy`.
   - Prompt metadata is tracked in `data/steering_state/progress.json`.

4. Vector extraction
   - `run_refusal_extraction.py` reloads all cached activations.
   - It selects a layer.
   - It computes:
     - a native-scale difference-in-means vector
     - an optional logistic-regression probe vector
   - Diagnostics are printed using Cohen's d and ROC AUC.

5. Vector packaging
   - Results are saved to `data/steering_vectors.npz`.
   - Current expected keys include:
     - `diff_of_means`
     - `probe`
     - `optimal_layer`

6. Intervention
   - `train_probe.py` loads `data/steering_vectors.npz`.
   - It reloads the base instruct model.
   - It demonstrates three scenarios:
     - baseline generation with no intervention
     - strong additive steering at the chosen residual hook
     - projection removal across all layers via attention and MLP outputs

### Legacy Flow

1. `legacy/run_steering_extraction.py` loads an abliterated model through `setup_model.py`.
2. It extracts only one residual layer per prompt.
3. It saves `[d_model]` activations per prompt.
4. It recomputes a steering vector from accumulated safe vs unsafe activations.
5. `steering/apply_steering.py` can then inject that vector at a chosen
   `hook_resid_post`.
6. `tests/test_steering_intervention.py` still reflects this older framing.

## File Map And Responsibilities

## Root Docs And Config

- `README.md`
  - Early setup and smoke-test instructions.
  - Focuses on `setup_model.py` and `legacy/extract_h_vectors.py`.
  - Useful for environment setup, but it does not describe the current refusal
    extraction and projection-removal workflow.

- `legacy/guide/conceptual.md`
  - Theory note on linear representation, probes, difference-in-means, and activation
    steering.
  - Still useful as background intuition.
  - Partly stale because it describes unit normalization as the standard approach,
    while the code now preserves native vector scale.

- `requirements.txt`
  - Core Python dependencies.
  - PyTorch is intentionally installed separately.

- `.env.template`
  - Environment template for model access and device selection.

- `.GITIGNORE`
  - Intended to ignore `.env` and `data/`.
  - Important because much of the actual runtime state lives under `data/`.

## Model Loading And Shared Helpers

- `setup_model.py`
  - General `HookedTransformer` loader.
  - Can load a normal model or wrap an abliterated HF checkpoint into a base
    architecture.
  - Honors environment variables only when explicit function arguments are absent.
  - Still important for legacy flows and tests.

- `model_utils.py`
  - Thin convenience layer over `setup_model.py`.
  - Also contains shared helpers for:
    - short generation
    - running with cache
    - finding residual keys
    - retrieving final-token activations

## Data Formatting

- `data/format_xstest.py`
  - XSTest ingestion and formatting utilities.
  - Supports two modes:
    - full dataset export
    - matched safe/unsafe prompt pairing
  - The current main extraction path uses the full-dataset CSV.
  - The matched-pair path remains for older tests and experiments.

- `legacy/data_prep.py`
  - Earlier helper for building paired prompt dataloaders using chat templates.
  - Mostly aligned with the older demo or prototype path rather than the current
    resumable XSTest pipeline.

## Early Extraction Prototype

- `legacy/extract_h_vectors.py`
  - Small demo script that runs a dataloader through the model, caches residuals, and
    writes final-token vectors to `h_vectors.pkl`.
  - Best understood as an initial proof of concept rather than the main production
    pipeline.

## Activation Collection

- `activations/cache_activations.py`
  - Core prompt-level activation capture utility.
  - `generate_and_cache_prompt(...)`:
    - tokenizes the prompt
    - runs `run_with_cache`
    - extracts prompt-final `resid_post` vectors for all layers
    - frees cache memory
    - then generates a text response for logging
  - `get_activations_for_prompts(...)`:
    - batched helper for one chosen layer
    - more aligned with the older single-layer flow

## Probe Training

- `probe/train_probe.py`
  - The actual probe-training module.
  - Trains a logistic regression classifier on safe vs unsafe activations.
  - Returns the trained probe plus simple metrics and a classification report.

## Steering Math And Diagnostics

- `steering/steering_utils.py`
  - Difference-in-means extraction.
  - Probe-vector conversion.
  - Diagnostics based on scalar projection scores.
  - Important current behavior:
    - `steering_vector_diff_of_means(...)` preserves native scale
    - this codifies the project's move away from unit-normalized steering vectors

- `steering/apply_steering.py`
  - Reusable single-layer activation-addition hooks.
  - Adds `alpha * steer_vec` only at the last token position of
    `blocks.{layer}.hook_resid_post`.
  - This is the clean reusable library version of classic activation addition.
  - It is not the "extreme projection removal" implementation.

## Main Extraction Pipelines

- `legacy/run_steering_extraction.py`
  - Older incremental extraction script for the abliterated-model path.
  - Stores one `[d_model]` activation per prompt for one chosen layer.
  - Uses:
    - `model_utils.load_model(...)`
    - `data/format_xstest.py`
    - `activations/cache_activations.py`
    - `probe/train_probe.py`
    - `steering/steering_utils.py`
  - Saves:
    - `data/steering_state/...`
    - `data/steering_vectors.npz`

- `run_refusal_extraction.py`
  - Current extraction script for the base instruct model.
  - Main modern entry point for building the refusal vector.
  - Key differences from `legacy/run_steering_extraction.py`:
    - loads the instruct model directly instead of using `setup_model.py`
    - caches all layers per prompt instead of one layer
    - saves `optimal_layer` into `data/steering_vectors.npz`
  - This is the script that best reflects the project's current direction.

- `legacy/extract_refusal.py`
  - Separate experiment based on weight-difference SVD between safe and abliterated
    model weights.
  - Pulls MLP `down_proj` weights at one hardcoded layer and extracts the top singular
    vector.
  - Not part of the main activation-based pipeline.
  - Best treated as an isolated exploratory artifact from the abliteration-era framing.

## Intervention Harness

- `train_probe.py`
  - Despite its name, this file does not train probes.
  - It is an inference-time intervention script and test harness.
  - Loads vectors from `data/steering_vectors.npz`.
  - Defines:
    - a probe-like guardrail scaffold
    - a plain additive steering hook
    - an all-layer projection-removal hook set
  - Runs a hardcoded unsafe prompt through:
    - baseline generation
    - projection removal
    - strong additive steering
  - This file currently acts as the clearest demo of the final "extreme intervention"
    idea.

## Tests

- `tests/test_model_integration.py`
  - Manual integration sanity check.
  - Loads a model, generates one harmful prompt response, and inspects residual keys
    and activation shape.

- `tests/test_steering_extraction.py`
  - Older end-to-end smoke test using matched XSTest pairs.
  - Caches all-layer activations, trains a probe, prints diagnostics, and writes
    artifacts under `data/`.
  - Useful as a compact extraction smoke test, but not the authoritative current flow.

- `tests/test_steering_intervention.py`
  - Older intervention test for subtractive steering on an abliterated-model path.
  - Recomputes a difference-in-means vector from cached activations and applies a
    single-layer residual hook.
  - Still useful for the classic activation-addition code path.
  - Does not cover the all-layer projection-removal intervention.

## Notebooks

- `eda.ipynb`
  - Small exploratory notebook for inspecting `data/xstest_full.csv`.
  - Mostly dataset inspection and prompt browsing.

- `llm-endpoint.ipynb`
  - Side experiment that queries an external LLM endpoint directly.
  - Not part of the steering pipeline.
  - Best treated as a standalone experiment about external model behavior rather than
    this repo's core logic.

## Data And Artifact Flow

Most runtime state is expected under `data/`, even though that directory is not the
main source-code area.

### Source-like file under `data/`

- `data/format_xstest.py`
  - Actual Python utility module used by the scripts.

### Generated or cached artifacts

- `data/xstest_full.csv`
  - Full XSTest export with formatted prompts.

- `data/matched_xstest_prompts.csv`
  - Older paired safe/unsafe export.

- `data/matched_xstest_with_responses.csv`
  - Smoke-test output pairing prompts with model generations.

- `data/xstest_all_layers_activations.npz`
  - All-layer activations from the older extraction smoke test.

- `data/xstest_layer{N}_activations.npz`
  - One chosen layer from the older smoke test.

- `data/steering_state/progress.json`
  - Persistent registry of processed prompts.

- `data/steering_state/safe/{sha}.npy`
  - Safe prompt activations.

- `data/steering_state/unsafe/{sha}.npy`
  - Unsafe prompt activations.

- `data/steering_vectors.npz`
  - Main packaged vector artifact for later intervention.
  - Depending on which script produced it, it may contain:
    - `diff_of_means`
    - `probe`
    - optionally `optimal_layer`

- `data/svd_refusal_vector.pt`
  - Output of the separate SVD experiment in `legacy/extract_refusal.py`.

- `h_vectors.pkl`
  - Output of the early prototype in `legacy/extract_h_vectors.py`.

## Recommended Mental Model

If you are trying to understand the repo quickly, the safest summary is:

- `run_refusal_extraction.py` is the current extractor.
- `train_probe.py` is the current intervention demo.
- `steering/` contains reusable math and single-layer hook utilities.
- `probe/train_probe.py` is the actual probe trainer.
- `legacy/run_steering_extraction.py`, `legacy/extract_h_vectors.py`, and some tests
  preserve older paths and are still informative, but they are not the clearest
  description of the current project direction.

## Known Mismatches And Stale Areas

- `train_probe.py` is misnamed.
  - It is not a training script.
  - It is an intervention and demo script.

- `legacy/extract_refusal.py` sounds like the main refusal extractor, but it is
  actually a separate SVD-on-weights experiment.

- `run_refusal_extraction.py` still has a docstring copied from the older extraction
  script in places.

- `README.md` mostly describes the early smoke-test flow, not the current preferred
  refusal-extraction pipeline.

- `legacy/guide/conceptual.md` still presents unit normalization as standard, while
  the current code explicitly preserves native vector scale to avoid LayerNorm collapse.

- Historical project memory says layer selection should use separability metrics such as
  Cohen's d and ROC AUC. Current code in `run_refusal_extraction.py` selects the layer
  by raw mean-difference norm, then reports diagnostics after the fact.

- Historical project memory describes probe-triggered dynamic steering as a core goal.
  In current code, the conditional guardrail idea exists as a scaffold, but the most
  fully realized intervention is the always-on projection-removal path.

- Historical descriptions sometimes say the final intervention hooks every attention
  head output. Current code hooks every block's attention output and MLP output.

- The tests mostly validate older additive-steering flows and model integration. They
  do not comprehensively exercise the final projection-removal path in `train_probe.py`.

## Suggested Starting Points For Future Work

If returning to this project later, the most useful files to open first are:

1. `MEMORY.md`
2. `run_refusal_extraction.py`
3. `train_probe.py`
4. `steering/steering_utils.py`
5. `activations/cache_activations.py`
6. `data/format_xstest.py`

If trying to modernize the repo, the highest-value cleanup would likely be:

1. rename `train_probe.py` to match its actual role
2. align `README.md` with the current refusal-extraction pipeline
3. make the layer-selection criterion in `run_refusal_extraction.py` match the stated
   project logic
4. add a proper test for projection removal across all layers
