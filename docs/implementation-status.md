# Implementation status

The committed notebooks implement and document:

- Exploratory analysis of clicks, popularity, category preference, article length, co-occurrence, and inter-click timing.
- ItemCF article-to-article similarity and recall.
- UserCF user-to-user similarity and recall.
- FAISS-based article-embedding similarity retrieval.
- Popularity- and metadata-based cold-start candidate handling.
- Candidate-recall evaluation helpers and recall-result merging.

The original notebooks do not implement ranking. The new `src/pipeline.py` provides five retrieval channels (ItemCF, UserCF, content, learned two-tower and popularity), 31 ranking features, LightGBM Ranker and Classifier, content-based DIN, five user-grouped ranking folds, equal-weight blending and OOF logistic-regression stacking. `requirements-ranking.txt` supplies additional dependencies.

Limits: the dual tower pools history and does not encode temporal order; FAISS IndexFlatIP remains exact search; ranking OOF shares fixed outer-training-history retrieval. The new local validation is a sample experiment, not a 250,000-user run or replication of the historical metrics. No distributed index or IVF+PQ deployment is implemented. Refer to `docs/interview-alignment.md` before describing these components in an interview.

`src/study.py` adds a separate temporal study: past-only retrieval snapshots, ranking labels preceding independent validation requests, publication filtering, multi-seed 10,000-user evaluation, five retrieval-route ablations, three feature-group ablations, and sparse-history subgroup metrics. `scripts/summarize_study.py` generates the measured report and a source/data/runtime evidence manifest. This study uses a fixed validation retrieval snapshot, so freshness remains a measured limitation rather than a completed rolling-update feature.
