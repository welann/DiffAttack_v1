# Repository Guidelines

## Project Structure & Module Organization
- Core orchestration lives in `main.py`, which drives latent-space optimization in `diff_latent_attack.py` (use `diff_latent_attack-0.9.0.py` when pinning `diffusers==0.9.0` for TPAMI replication).
- Auxiliary modules sit in focused packages: `Finegrained_model/` (model loaders), `torch_nets/` (converted Inception variants), `dataset_caption/` (label metadata), and `pytorch_fid/` (Inception + FID stats).
- Demo assets in `demo/` supply smoke-test images and labels; push new sample data there and keep large datasets external.
- Store downloaded checkpoints under `pretrained_models/` (ignored by git) and keep outputs in run-specific folders.

## Build, Test & Development Commands
- Create an env and install deps: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Generate adversarial examples: `python main.py --model_name inception --save_dir runs/exp1 --images_root demo/images --label_path demo/labels.txt`.
- Evaluate transferability: `python main.py --is_test True --save_dir runs/exp1/eval --images_root runs/exp1 --label_path demo/labels.txt`.
- Compare perceptual metrics: `python pytorch_fid/fid_score.py demo/images runs/exp1/adv`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, `snake_case` for functions/args, and `CamelCase` for classes; keep import blocks sorted as in existing scripts.
- Add CLI knobs via `argparse` in `main.py` with clear help text, and reuse helpers in `utils.py`.
- Seed randomness through `seed_torch` when introducing new stochastic procedures to keep outputs reproducible.

## Testing Guidelines
- Run the demo command pipeline before every PR to confirm optimization, masking, and logging still work; inspect `runs/<exp>/log.txt` and generated grids.
- For new datasets or checkpoints, document required flags and drop a minimal sample pair under `demo/` for quick validation.
- When altering evaluation metrics or diffusion schedules, capture before/after FID or accuracy numbers in the PR description.

## Commit & Pull Request Guidelines
- Use concise, present-tense commit titles (e.g., `Support diffusers 0.30.3`) and focus each commit on one logical change.
- Reference linked issues in commits or PRs; describe dataset/model impacts and paste the exact verification commands you executed.
- PRs should attach representative visuals or metric logs and call out any new artifacts that contributors must download manually.
- Keep generated images, large datasets, and private weights out of version control; rely on `.gitignore` and external storage.

## Model Weights & Security
- Stage third-party checkpoints inside `pretrained_models/` and never commit licensed or proprietary weights.
- Document the source and license of new models, and scrub API keys or tokens from scripts—load them via environment variables when required.
