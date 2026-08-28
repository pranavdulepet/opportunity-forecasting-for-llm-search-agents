# Opportunity Forecasting for LLM Search Agents

Search agents face the decision of whether to continue searching or stop and commit to the best option seen so far. We study how Large Language Models (LLMs) can predict how much more the reward can improve under a fixed search model. Source trajectories use deterministic decoding; Monte Carlo labels use sampled continuations from the same model and action prompt. We evaluate the forecasts on aligned held-out deterministic replay streams in two ways: either stopping a trajectory when predicted improvement is not worth its cost, or expanding the search thread with the greatest estimated remaining reward increase.

## Method

The experimental pipeline has four stages:

1. Run a fixed search model to collect decision points from WebShop and Paper
   Search trajectories.
2. Run six Monte Carlo continuations from each decision point and record the
   improvement over the best reward already observed.
3. Train forecasting heads.
4. Evaluate each forecast on held-out replay streams using stopping and
   budgeted-expansion rules.

The learned methods are:

- **ZOIB regression:** a zero-one-inflated Beta distribution over reward improvement.
- **Support-aware ZOIB:** a ZOIB distribution scaled by the possible remaining
  reward, `1 - current_best`.
- **Scalar head:** direct regression of the mean continuation reward improvement.
- **Gaussian head:** a Gaussian model of continuation reward improvement mean and variance.

## Domains And Data

| Domain | Reward mode | Train | Development | Test |
|---|---|---:|---:|---:|
| WebShop | `product_page_buy_now_current_options` | 19,967 | 1,997 | 2,007 |
| Paper Search | `paper_page_litsearch_webshop_relevance_v4` | 19,967 | 1,997 | 2,007 |

WebShop scores an opened product page using the benchmark buy-now reward for
the options currently selected in the session. Paper Search follows
the same search, open-candidate, and commit-to-best structure for scientific
papers, using qrels and title/abstract relevance.

## Installation

Clone the repository and create the training environment:

```bash
git clone https://github.com/pranavdulepet/opportunity-forecasting-for-llm-search-agents.git
cd opportunity-forecasting-for-llm-search-agents

conda env create -f environments/training.yml
conda activate opportunity-forecasting
python -m pip install -e . --no-deps
```

Download and validate the canonical experiment data, then use `Qwen/Qwen2.5-7B-Instruct`:

```bash
python -m opportunity_forecasting download-data
python -m opportunity_forecasting validate-data
python -m opportunity_forecasting prepare-model
```

## Training And Evaluation

The full experiment is launched as Slurm jobs:

```bash
python -m opportunity_forecasting experiment \
  --scheduler slurm \
  --start-from labels \
  --run-root runs/paper \
  --gpu-partition GPU_PARTITION \
  --cpu-partition CPU_PARTITION
```

This command runs the paper pipeline from the canonical labels through training,
prediction, evaluation, and result materialization.

Use a dry run to inspect every generated command and dependency without
submitting jobs:

```bash
python -m opportunity_forecasting experiment \
  --scheduler slurm \
  --start-from labels \
  --run-root runs/paper \
  --dry-run
```

To resume an interrupted experiment, `--start-from checkpoints` uses trained
heads already present in the selected run directory, while `--start-from
predictions` resumes from its completed prediction files.

## Paper Figures And Tables

Generated files are written to `paper_outputs/`. Individual outputs can be
rendered with:

| Output | Command |
|---|---|
| Forecasting pipeline | `python -m opportunity_forecasting figure overview` |
| Budgeted expansion | `python -m opportunity_forecasting figure budgeted-expansion` |
| Absolute reward | `python -m opportunity_forecasting figure absolute-reward` |
| Stopping frontiers | `python -m opportunity_forecasting figure stopping` |
| Search-value profile | `python -m opportunity_forecasting figure search-value` |
| Tables | `python -m opportunity_forecasting tables` |

For a completed experiment run, initialize new figure and table sources with:

```bash
python -m opportunity_forecasting summarize --run-root runs/paper
```

## Rebuilding The Supervised Dataset

The repository includes the data-generation implementation used before model
training:

- `opportunity_forecasting.data.trajectories` generates decision-point streams.
- `opportunity_forecasting.data.label_webshop` labels WebShop states.
- `opportunity_forecasting.data.label_paper_search` labels Paper Search states.
- `opportunity_forecasting.data.merge_labels` validates and merges parallel
  shards.
- `opportunity_forecasting.data.splits` creates goal-disjoint splits.

Generation settings and prompt hashes are recorded under each domain's
`trajectory_generation` and `label_generation` entries in `configs/paper.json`.

```bash
conda env create -f environments/webshop.yml
conda activate opportunity-forecasting-webshop
python -m pip install -e . --no-deps
python -m opportunity_forecasting prepare-webshop --download-data --build-index
```

Paper Search combines LitSearch with SciDocs, SciFact, NFCorpus, and TREC-COVID.
The data includes the exact query, corpus, and qrel export used by
the paper.

## Repository Structure

```text
opportunity_forecasting/
  data/          environments, trajectory generation, labels, and validation
  models/        forecasting distributions, training, and inference
  evaluation/    stopping, allocation, controls, and diagnostics
  figures/       figure, table, and result materialization
  experiments/   Slurm orchestration and local evaluation
configs/          canonical experiment manifest
data/             split metadata and downloaded canonical data
environments/     analysis, training, and WebShop environments
requirements/     pinned dependency sets
results/          numeric sources for reported figures and tables
tests/            scientific invariants and pipeline tests
```

## License

Original code is released under the MIT License. Third-party models, datasets,
and bundled fonts retain their original licenses and terms.
