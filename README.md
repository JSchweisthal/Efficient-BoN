# Efficient Best-of-N policy evaluation for inference-time alignment

This repository contains the minimal implementation and paper artifacts for
off-policy evaluation and safe inference-time budget selection for Best-of-N
policies. It covers exactly two settings: controlled synthetic experiments and
a realistic LLM setting on GSM8K.

## Structure

```text
bon/                         BoN estimators, paired inference and budget selection
dgp/                         synthetic DGP; GSM8K prompts, generation and reward scorers
experiments/                 per-seed runners and cross-seed aggregation for both settings
tests/                       exact estimator identities and the selection fallback
plotting/figures.ipynb       Figures 1–12 and Tables 1–6 in paper order
plotting/data/               cross-seed tables written by `aggregate`, read by the notebook
results/                     per-seed records written by the runners
figures/                     paper figures written by the notebook
tables/                      paper tables (CSV and HTML) written by the notebook
run_experiment.py            command-line entry point
```

## Experiment settings

### Synthetic experiments

The controlled rare-tail DGP separates score error from reward-function error.
Seventeen cases vary score error, reward-function error, validation size `n` and
auxiliary pool size `M` one at a time (300 seeds each). Experiments study
estimator robustness, validation sample size, finite-pool efficiency, and
selection by Value-max and Improvement-LCB.

### Realistic LLM setting

Reference models:

- Gemma-2-2B-IT (`google/gemma-2-2b-it`)
- Llama-3.2-3B-Instruct (`meta-llama/Llama-3.2-3B-Instruct`)
- Qwen3.5-4B (`Qwen/Qwen3.5-4B`)

Reward models:

- OASST-RM-2.1-Pythia-1.4B (`OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5`)
- Skywork-Reward-V2-Qwen3-1.7B (`Skywork/Skywork-Reward-V2-Qwen3-1.7B`)

Response generation and reward scoring need GPUs; the code for both is in
`dgp/`. Everything after scoring runs on CPU from stored scores and correctness
labels: the score-spline outcome regression, the fixed-bank design (100 seeds,
selection scored on held-out test banks) and the IID calibration design (500
seeds, `M = 512`).

## Run

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
```

### Synthetic experiments

```bash
python run_experiment.py synthetic                  # all 17 cases x 300 seeds
python run_experiment.py aggregate synthetic        # writes plotting/data/synthetic/
```

Records are resumable and can be split across processes by seed, e.g.
`--seeds 0:50`, `--seeds 50:100`, ….

### GSM8K

For each reference model, `--outcomes` and `--scores` point to directories with
`train/`, `validation/` and `test/` stores. A store holds Parquet files with
columns `question_id`, `sample_id` and either `score` (under `rewards/`) or the
boolean exact-match label `correct` (under `outcomes/`); a score store may carry
a `manifest.json` whose `model_id` is checked against `--selector`.

```bash
python run_experiment.py llm --model gemma --selector oasst \
  --scores SCORES/gemma/oasst --outcomes OUTCOMES/gemma
# repeat for every model in {gemma, llama, qwen} and selector in {oasst, skywork}
python run_experiment.py aggregate llm              # writes plotting/data/real/
```

Table 3 uses independently trained reward functions (TF-IDF logistic and
embedding LightGBM). Their validation predictions are passed as a score-format
store:

```bash
python run_experiment.py llm --model gemma --selector oasst \
  --scores SCORES/gemma/oasst --outcomes OUTCOMES/gemma \
  --nuisance-name tfidf_logistic --nuisance PREDICTIONS/gemma/tfidf_logistic
```

`aggregate llm` then also writes `plotting/data/independent_nuisance/`.

### Figures and tables

After aggregating both settings, open `plotting/figures.ipynb` and run all cells;
it writes Figures 1–12 to `figures/` and Tables 1–6 to `tables/`.
