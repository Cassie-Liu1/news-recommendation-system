"""Execute the retrieval notebook locally without a Jupyter server.

Installation cells and unused ranking imports are skipped. Embedding search
queries only debug-history articles against the complete article catalogue.
The original notebook is preserved; executed cells are saved under outputs.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
notebook = json.loads((ROOT / 'notebooks/recommendation_model.ipynb').read_text())
namespace = {'__name__': '__main__'}
started = time.time()

class Tee(io.StringIO):
    def write(self, value):
        import sys
        sys.__stdout__.write(value)
        sys.__stdout__.flush()
        return super().write(value)

embedding_cell = '''
faiss.omp_set_num_threads(4)
item_emb_df = pd.read_csv(data_path + 'articles_emb.csv', dtype=np.float32)
ids = item_emb_df['article_id'].to_numpy(dtype=np.int64)
cols = [c for c in item_emb_df if c.startswith('emb_')]
vectors = np.ascontiguousarray(item_emb_df[cols].to_numpy(), dtype=np.float32)
faiss.normalize_L2(vectors)
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)
query_positions = np.flatnonzero(np.isin(ids, all_click_df.click_article_id.unique()))
emb_i2i_sim = collections.defaultdict(dict)
for start in range(0, len(query_positions), 256):
    positions = query_positions[start:start + 256]
    scores, neighbors = index.search(vectors[positions], 151)
    for pos, score_row, neighbor_row in zip(positions, scores, neighbors):
        source = int(ids[pos])
        emb_i2i_sim[source] = {int(ids[j]): float(s) for s, j in zip(score_row, neighbor_row)
                              if j >= 0 and int(ids[j]) != source}
pickle.dump(emb_i2i_sim, open(save_path + 'emb_i2i_sim.pkl', 'wb'))
print('Embedding catalogue:', len(ids), 'history queries:', len(query_positions))
del item_emb_df, vectors, index
'''

try:
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        cell['outputs'] = []
        cell['execution_count'] = None
        if i in (1, 3, 13):
            cell['source'] = ['# Skipped: installation or unused embedding-dictionary cache.\n']
            continue
        if i == 4:
            source = '\n'.join(line for line in source.splitlines()
                               if not line.startswith(('import lightgbm', 'from gensim')))
            source += '\nnp.random.seed(42)\nrandom.seed(42)\n'
        if i == 49:
            source = embedding_cell
        if i == 51:
            source = source.replace('i2i_sim[i].items()', 'i2i_sim.get(i, {}).items()')
        cell['source'] = source.splitlines(keepends=True)
        print(f'Executing cell {i}', flush=True)
        capture = Tee()
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            exec(compile(source, f'notebook-cell-{i}', 'exec'), namespace)
        cell['execution_count'] = i + 1
        cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': capture.getvalue()}]
    np = namespace['np']
    truth = dict(zip(namespace['trn_last_click_df'].user_id,
                     namespace['trn_last_click_df'].click_article_id))
    merged = namespace['final_recall_items_dict_rank']
    metrics = {}
    for k in (5, 10):
        hit = 0
        ndcg = 0.0
        for user, target in truth.items():
            ranked = [item for item, _ in merged.get(user, [])[:k]]
            if target in ranked:
                hit += 1
                ndcg += 1 / np.log2(ranked.index(target) + 2)
        metrics[f'HitRate@{k}'] = hit / len(truth)
        metrics[f'NDCG@{k}'] = ndcg / len(truth)
    result = {'sample_users': 1000, 'seed': 42, 'evaluation_users': len(truth),
              'elapsed_seconds': round(time.time() - started, 2), 'metrics': metrics,
              'scope': 'debug retrieval; not ranking ensemble or full-population evaluation'}
    (OUT / 'local_run_metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
except Exception:
    traceback.print_exc()
    raise
finally:
    (OUT / 'recommendation_model.executed.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
