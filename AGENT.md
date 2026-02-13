# Agent Instructions

## Python Environment

Use the following Conda environment for this repository:

```bash
conda activate lerobot
```

If the named environment is unavailable, use the absolute path:

```bash
conda activate /llm_jzm/cache/conda_env/lerobot
```

If using `conda activate lerobot`:
- Treat it as personal machine mode.
- Do not set shared-machine Hugging Face cache or token overrides.

If using `conda activate /llm_jzm/cache/conda_env/lerobot`:
- Treat it as shared machine mode.
- Hugging Face cache is not in the default location.
- Set Hugging Face token to the shared token for runs on this machine.
- Set Wandb token to hte shared token for runs on this machine

Use:

```bash
export HF_HOME=/llm_jzm/cache/huggingface/
export HF_TOKEN=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
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY=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
wandb login --relogin "$WANDB_API_KEY"
```