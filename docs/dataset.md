# Dataset

## Source files

The project uses four source tables:

| File | Purpose | Availability in this repository |
| --- | --- | --- |
| `train_click_log.csv` | Training-user click events | Excluded |
| `testA_click_log.csv` | Evaluation-user click events | Excluded |
| `articles.csv` | Article metadata | Excluded |
| `articles_emb.csv` | Article content embeddings | Excluded |

The original raw files are excluded from version control because they are shared through a restricted Drive folder and include large files. In particular, the article-embedding CSV is approximately 973 MB.

## Verified scale and schema

The committed notebooks report the following:

- 200,000 distinct training users and 50,000 distinct Test-A users.
- Training and evaluation user IDs do not overlap.
- The article embedding table has 364,047 rows and 251 columns: an article identifier plus 249 embedding dimensions and associated metadata.
- Click-log fields include `user_id`, `click_article_id`, `click_timestamp`, `click_environment`, `click_deviceGroup`, `click_os`, `click_country`, `click_region`, and `click_referrer_type`.
- Article metadata includes article ID, category, creation timestamp, and word count.

## Add local data

Create a local `data/` directory (ignored by Git) and place the four source CSVs inside it. Update the data-path cell in each notebook before execution.

Never commit raw click logs, embedding files, personal Drive paths, or generated model binaries to a public repository.
