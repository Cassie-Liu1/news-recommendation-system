# News Recommendation System

An offline, multi-stage recommendation system that predicts each user's next news click and returns a ranked Top-5 list from more than 360,000 articles.

## Rolling retrieval: latest comparison

The latest three-seed, 10,000-user temporal comparison refreshes all retrieval routes and source-derived features in four chronological validation bins. With the same sampled users, time split and trained ranker, mean HitRate@5 increases from **1.700% to 2.067%** (+0.367 percentage points), and candidate coverage from **2.184% to 2.834%**. This is a batched offline comparison, not production performance or a significance claim. See the [rolling retrieval report](docs/rolling-study.md).

## Temporal study: fixed-snapshot benchmark

Three 10,000-user samples (seeds 42, 123, 2026), each with 8,000 training users and approximately 2,000 later validation users. All requests use past-only retrieval snapshots and already-published candidate articles. Mean HitRate@5: **1.70%**, seed standard deviation **0.17 percentage points**. Candidate coverage: **2.18%**.

Removing content retrieval lowers mean HitRate@5 to **0.42%**; removing time features lowers it to **0.48%**. The fixed retrieval snapshot becomes stale across later news requests, so retrieval freshness is a key remaining limitation. These results do not establish online performance or reproduce the historical 61% claim. Full metrics, label audits, group results and limitations are in the [temporal experiment report](docs/temporal-study.md).

## Earlier random-user experiment

The later temporal validation and ablation study uses a different protocol; see [temporal experiment report](docs/temporal-study.md). The table below is the earlier random-user split, not evidence for temporal generalisation.

Seed 42, 1,000 sampled users, 800 training users and 200 disjoint validation users; five user-grouped ranking folds and two neural-training epochs. Runtime: 40.4 seconds on the local run. All models share the same validation candidates (31.5% positive coverage).

| Method | HitRate@5 | NDCG@5 |
| --- | ---: | ---: |
| Recall-only RRF | 0.090 | 0.0601 |
| LightGBM Ranker | 0.210 | 0.1488 |
| LightGBM Classifier | 0.190 | 0.1299 |
| DIN | 0.110 | 0.0689 |
| Equal-weight blend | 0.205 | 0.1393 |
| OOF Stacking | 0.210 | 0.1421 |

Stacking improves HitRate@5 by 12 percentage points over this recall baseline, but does not beat the single Ranker. Its conditional GAUC is 0.9528 over only the 63 users with recalled positives and negative candidates; it is not a whole-population AUC. See [experiment notes](docs/interview-alignment.md). The former notebook's 1.6% debug result uses a different protocol and is not directly comparable.

## Problem

The project uses click logs from a news application to predict each user's held-out final article click. It addresses a cold-start evaluation setting: training and evaluation users are completely disjoint, and some evaluation users have only one interaction. The solution therefore avoids relying on user-ID memorisation and instead uses short-term click sequences, content similarity, and popularity signals.

**Source-data scale:** 200,000 training users and 50,000 Test-A users, 360K+ articles, and 249-dimensional article embeddings. Completed experiments use sampled users; a full 250,000-user recommendation run has not been verified.

## System design

```text
Click logs + article metadata + article embeddings
                    │
                    ▼
        EDA and feature hypothesis generation
                    │
                    ▼
  Implemented candidate retrieval
  ├── ItemCF: co-click article similarity
  ├── UserCF: similar-user preferences
  ├── Embedding retrieval: FAISS nearest neighbours
  └── Popularity fallback: cold-start users
                    │
                    ▼
              Ranked Top-5 recommendations
```

## What I analysed

- User activity, repeat clicks, click environments, and inter-click timing
- Article popularity and long-tail behaviour
- Category breadth, topic preferences, and preferred article length
- Sequential co-occurrence and short-term interest continuity

These findings informed the feature design and justified cold-start, recency, collaborative-filtering, and content-based retrieval strategies.

## Feature engineering

The notebooks focus on retrieval. `src/pipeline.py` adds 31 features covering activity, raw timestamp intervals in days/seconds, popularity, category and word-count preference, content similarity, and each retrieval channel's presence, rank and score. Neither user ID nor article ID is a ranking input.

## Model evaluation

The legacy notebook uses a 1,000-user debug sample. The new pipeline samples 1,000 eligible users, holds out each final click, then assigns 80% of users to training and 20% to independent validation. Retrieval statistics and the learned dual tower use only training histories; final-click targets never build retrieval. Five user-grouped folds produce out-of-fold predictions from LightGBM Ranker, LightGBM Classifier and a content-based DIN implementation. A logistic-regression stacker fits only these OOF predictions. Inner folds share a fixed retrieval model built from outer training histories; this is not fully nested retrieval cross-validation.

No positives are injected into candidate sets. Missing candidates count as misses in HitRate and NDCG. GAUC is the equal-weight average over users with both positive and negative candidates, and its eligible-user count is reported. Training and validation predictions, model files and Top-5 outputs are saved locally. Small-sample results do not establish full-population performance.

## Tech stack

Python · Pandas · NumPy · Matplotlib · Seaborn · FAISS · ItemCF · UserCF · Google Colab

## Repository structure

`src/` contains the executable retrieval, ranking and temporal-study pipelines;
`scripts/` contains local runners and report generation; `tests/` contains
contract checks; `docs/` contains measured experiment reports.

Owner-supplied legacy notebooks and extracted EDA images remain local while
their original authorship and publication permissions are unverified.
Raw click logs, embeddings and trained model files are not distributed.

## Run locally

Install `requirements-ranking.txt` in a Python virtual environment.

For the new complete retrieval/ranking experiment, place the four CSVs in `data/`, then run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-ranking.txt
python src/pipeline.py --users 1000 --epochs 2 --seed 42
python -m unittest discover -s tests -p test_pipeline.py -v
```

Run the temporal study and regenerate its evidence-backed report:

```bash
python src/study.py --users 10000 --seeds 42 123 2026 --output temporal_10000_v1
python scripts/summarize_study.py --study temporal_10000_v1
python -m unittest discover -s tests -v
```

Use a new `--output` directory for a repeat run so earlier measurements are preserved. The temporal study includes expanding past-only training snapshots, disjoint chronological validation, raw label-availability/repeat audits, per-channel retraining ablations, feature-group retraining ablations, and history-length subgroups. It deliberately keeps its results separate from the earlier random split.

The script queries all 364,047 article embeddings using FAISS exact inner-product search. The learned retrieval branch is a YouTubeDNN-style mean-history two-tower model trained with sampled softmax, not a temporal sequence encoder or a reproduction of the original industrial system. DIN uses candidate-conditioned attention over frozen content vectors. Training and validation users are disjoint. The default run generates validation-user predictions, not a competition submission for Test-A.

The original dataset and generated model artifacts are excluded from version control. See [dataset documentation](docs/dataset.md), [implementation status](docs/implementation-status.md), and [interview-to-code mapping](docs/interview-alignment.md).

## Portfolio and interview note

This repository documents the project approach and results. Before publishing externally, only claim the modules, experiments, and results you personally completed or can explain and reproduce.
# Rolling retrieval comparison

Use the same sampled users, temporal split and fixed trained ranker while refreshing
retrieval in four chronological validation bins. Only observed training-user events
(including their final clicks once observed) enter the refreshed sources. Validation
users and targets never enter source statistics. This is batched causal refresh,
not a deployed real-time service; all routes and source-derived features refresh.

```sh
PYTHONPATH=/tmp/news-rec-runtime python src/study.py --users 10000 --seeds 42 123 2026 --rolling-bins 4 --output rolling_10000_v1
PYTHONPATH=/tmp/news-rec-runtime python scripts/summarize_study.py --study rolling_10000_v1 --report rolling-study.md
```

The temporary runtime path above describes this local execution; other machines
should install the project requirements into their own environment and omit it.
Use a new output name when repeating runs to preserve existing evidence.
