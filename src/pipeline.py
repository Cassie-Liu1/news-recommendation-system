"""Local, user-disjoint news retrieval and ranking experiment.

Candidates are never injected using labels. Frozen article vectors support
unseen users/items without ID learning.
"""
import argparse
import faulthandler
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import pickle
import time

# Keep independent OpenMP runtimes from creating competing worker pools on macOS.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

CHANNELS = ['itemcf', 'usercf', 'content', 'youtube_dnn', 'popular']
FEATURE_NAMES = ['history_clicks', 'history_span_days', 'mean_interclick_seconds',
                 'article_clicks', 'article_users', 'article_words', 'word_preference_gap',
                 'category_match_fraction', 'last_category_match', 'article_age_days',
                 'creation_gap_days', 'last_content_similarity', 'max_content_similarity',
                 'mean_content_similarity', 'channel_count', 'rrf_score'] + [
                     f'{channel}_{field}' for channel in CHANNELS for field in ['present', 'rank', 'score']]


class TwoTower(nn.Module):
    """YouTubeDNN-style mean-history user tower, sampled-softmax training.

    Pooling does not encode order: this is not a temporal sequence encoder.
    """
    def __init__(self, dim):
        super().__init__()
        self.user = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 32))
        self.item = nn.Linear(dim, 32)

    def forward(self, history, candidates):
        return (nn.functional.normalize(self.user(history), dim=-1),
                nn.functional.normalize(self.item(candidates), dim=-1))


class DIN(nn.Module):
    """Candidate-conditioned local activation over frozen content embeddings."""
    def __init__(self, dim, feature_dim):
        super().__init__()
        self.project = nn.Linear(dim, 16)
        self.attention = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.predict = nn.Sequential(nn.Linear(32 + feature_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, history, candidate, mask, features):
        h, c = self.project(history), self.project(candidate)
        q = c[:, None, :].expand_as(h)
        a = self.attention(torch.cat([h, q, h - q, h * q], dim=-1)).squeeze(-1)
        a = torch.softmax(a.masked_fill(~mask, -1e9), dim=1)
        interest = (a[:, :, None] * h).sum(dim=1)
        return self.predict(torch.cat([interest, c, features], dim=-1)).squeeze(-1)


class Retrieval:
    def __init__(self, histories, metadata, ids, vectors, seed, epochs, negative_positions=None):
        self.histories, self.metadata = histories, metadata
        self.article_info = {int(a): (int(c), float(t), int(w))
                             for a, c, t, w in metadata.reset_index().itertuples(index=False, name=None)}
        self.ids, self.vectors = ids, vectors
        self.position = {int(a): i for i, a in enumerate(ids)}
        self.popularity = Counter(a for rows in histories.values() for a, _ in rows)
        self.article_users = defaultdict(set)
        self.itemcf = defaultdict(Counter)
        self.user_items = {}
        for u, rows in histories.items():
            self.user_items[u] = {a for a, _ in rows}
            for a in self.user_items[u]:
                self.article_users[a].add(u)
            recent = rows[-20:]
            for i, (a, _) in enumerate(recent):
                for j, (b, _) in enumerate(recent):
                    if a != b:
                        self.itemcf[a][b] += 0.9 ** abs(i - j) / math.log1p(len(recent))
        for a, neighbors in self.itemcf.items():
            for b in neighbors:
                neighbors[b] /= math.sqrt(self.popularity[a] * self.popularity[b])
        self.unique_users = {a: len(us) for a, us in self.article_users.items()}
        self.content = faiss.IndexFlatIP(vectors.shape[1])
        self.content.add(vectors)
        torch.manual_seed(seed)
        self.tower = TwoTower(vectors.shape[1])
        x, y = [], []
        for rows in histories.values():
            for j in range(1, min(len(rows), 20)):
                x.append(self.mean(rows[max(0, j - 10):j]))
                y.append(self.position[rows[j][0]])
        rng = np.random.default_rng(seed)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        pool = np.arange(len(ids)) if negative_positions is None else negative_positions
        optimizer = torch.optim.Adam(self.tower.parameters(), lr=.002)
        for _ in range(epochs):
            for batch in np.array_split(rng.permutation(len(x)), max(1, math.ceil(len(x) / 128))):
                if not len(batch):
                    continue
                negatives = rng.choice(pool, (len(batch), 63))
                targets = y[batch]
                # Do not present a target as its own negative.
                overlap = negatives == targets[:, None]
                while overlap.any():
                    negatives[overlap] = rng.choice(pool, int(overlap.sum()))
                    overlap = negatives == targets[:, None]
                options = np.column_stack([targets, negatives])
                users, items = self.tower(torch.tensor(x[batch]), torch.tensor(vectors[options]))
                loss = nn.functional.cross_entropy(torch.einsum('bd,bkd->bk', users, items) / .1,
                                                    torch.zeros(len(batch), dtype=torch.long))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            learned = nn.functional.normalize(self.tower.item(torch.tensor(vectors)), dim=-1).numpy()
        self.learned = faiss.IndexFlatIP(learned.shape[1])
        self.learned.add(np.ascontiguousarray(learned))
        self.cache = {}
        self.history_features = {}

    def mean(self, rows):
        return self.vectors[[self.position[a] for a, _ in rows]].mean(axis=0)

    def retrieve(self, rows, count=30, user_id=None, as_of=None):
        seen = {a for a, _ in rows}
        channels = {c: defaultdict(float) for c in CHANNELS}
        for j, (a, _) in enumerate(rows[-10:]):
            w = .9 ** (len(rows[-10:]) - j - 1)
            for b, s in self.itemcf.get(a, Counter()).most_common(count):
                channels['itemcf'][b] += w * s
            if a not in self.cache:
                scores, positions = self.content.search(self.vectors[self.position[a]:self.position[a]+1], count + 10)
                self.cache[a] = [(int(self.ids[b]), float(s)) for b, s in zip(positions[0], scores[0]) if b >= 0]
            for b, s in self.cache[a]:
                channels['content'][b] += w * s
        neighbors = Counter()
        for a in seen:
            for u in self.article_users.get(a, set()):
                neighbors[u] += 1 / math.log1p(len(self.article_users[a]))
        for u in neighbors:
            neighbors[u] /= math.sqrt(len(seen) * len(self.user_items[u]))
        for u, s in neighbors.most_common(20):
            if u == user_id:
                continue
            for a in self.user_items[u]:
                channels['usercf'][a] += s
        with torch.no_grad():
            q = nn.functional.normalize(self.tower.user(torch.tensor(self.mean(rows[-10:]))[None]), dim=-1).numpy()
        scores, positions = self.learned.search(np.ascontiguousarray(q),
                                                max(count + len(seen), 256) if as_of is not None else count + len(seen))
        channels['youtube_dnn'].update({int(self.ids[b]): float(s) for b, s in zip(positions[0], scores[0]) if b >= 0})
        channels['popular'].update(self.popularity)
        result = {}
        for c, candidates in channels.items():
            ranked = sorted(((a, s) for a, s in candidates.items() if a not in seen
                             and (as_of is None or self.article_info[a][1] <= as_of)),
                            key=lambda p: (-p[1], p[0]))[:count]
            for rank, (a, score) in enumerate(ranked, 1):
                result.setdefault(a, {})[c] = (rank, float(score))
        return result

    def features(self, rows, article, sources):
        category, created, words = self.article_info[article]
        key = tuple(rows)
        if key not in self.history_features:
            information = [self.article_info[a] for a, _ in rows]
            cats = Counter(c for c, _, _ in information)
            times = np.asarray([t for _, t in rows], dtype=np.float64)
            self.history_features[key] = (
                cats, np.mean([w for _, _, w in information]),
                information[-1][0], information[-1][1], times[-1],
                (times[-1]-times[0])/86400000,
                np.diff(times).mean()/1000 if len(times)>1 else 0,
                self.vectors[[self.position[a] for a, _ in rows[-10:]]])
        cats, mean_words, last_category, last_created, last_time, span, interval, history_vectors = self.history_features[key]
        similarity = history_vectors @ self.vectors[self.position[article]]
        f = [len(rows), span, interval,
             self.popularity[article], self.unique_users.get(article, 0),
             words, abs(words-mean_words),
             cats[category]/len(rows), float(category==last_category),
             (last_time-created)/86400000,
             abs(created-last_created)/86400000,
             float(similarity[-1]), float(similarity.max()), float(similarity.mean()),
             len(sources), sum(1/(60+r) for r, _ in sources.values())]
        for c in CHANNELS:
            rank, score = sources.get(c, (0, 0))
            f.extend([float(c in sources), rank, score])
        return f


def candidate_table(retrieval, histories, labels, training):
    records, features, injected = [], [], 0
    for number, (u, rows) in enumerate(histories.items(), 1):
        candidates = retrieval.retrieve(rows)
        target = labels[u]
        # Missing positives remain missing, even during training. Injecting a
        # positive with zero channel features would expose a label-only shortcut.
        for article, sources in candidates.items():
            records.append((u, article, int(article == target)))
            features.append(retrieval.features(rows, article, sources))
        if number % 200 == 0:
            print('Candidate users processed:', number, '/', len(histories), flush=True)
    return pd.DataFrame(records, columns=['user_id', 'article_id', 'label']), np.asarray(features, dtype=np.float32), injected


def din_arrays(table, histories, retrieval, length=10):
    h = np.zeros((len(table), length, retrieval.vectors.shape[1]), dtype=np.float32)
    mask = np.zeros((len(table), length), dtype=bool)
    for i, (u, a, _) in enumerate(table.itertuples(index=False, name=None)):
        rows = histories[u][-length:]
        h[i, :len(rows)] = retrieval.vectors[[retrieval.position[b] for b, _ in rows]]
        mask[i, :len(rows)] = True
    c = retrieval.vectors[[retrieval.position[a] for a in table.article_id]]
    return h, c, mask


def din_predict(model, arrays, features):
    model.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(model(*(torch.tensor(a[i:i+512]) for a in arrays),
                                                  torch.tensor(features[i:i+512]))).numpy()
                               for i in range(0, len(features), 512)])


def evaluate(table, scores, users):
    ranked = table.assign(score=scores).sort_values(['user_id', 'score', 'article_id'], ascending=[True, False, True])
    groups = {u: g for u, g in ranked.groupby('user_id')}
    hit, ndcg, aucs = 0, 0., []
    for u in users:
        g = groups.get(u)
        if g is None:
            continue
        positives = np.flatnonzero(g.label.to_numpy())
        if len(positives):
            rank = int(positives[0])
            if rank < 5:
                hit += 1; ndcg += 1/math.log2(rank+2)
            if g.label.nunique() == 2:
                aucs.append(roc_auc_score(g.label, g.score))
    return {'HitRate@5': hit/len(users), 'NDCG@5': ndcg/len(users),
            'GAUC_recalled_users_equal_weight': float(np.mean(aucs)) if aucs else None,
            'GAUC_eligible_users': len(aucs), 'evaluation_users': len(users),
            'candidate_hit_rate': table.groupby('user_id').label.max().sum()/len(users)}


def run(args):
    started = time.time()
    root = Path(__file__).resolve().parents[1]
    out = root/'outputs'/'interview_alignment'; out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); faiss.omp_set_num_threads(1)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    data = root/'data'
    clicks = pd.read_csv(data/'train_click_log.csv').drop_duplicates(['user_id','click_article_id','click_timestamp'])
    test_users = pd.read_csv(data/'testA_click_log.csv', usecols=['user_id']).user_id.unique()
    assert not set(clicks.user_id.unique()) & set(test_users)
    eligible = clicks.groupby('user_id').size(); eligible = eligible[eligible >= 2].index.to_numpy()
    users = np.random.default_rng(args.seed).choice(eligible, min(args.users,len(eligible)), replace=False)
    clicks = clicks[clicks.user_id.isin(users)].sort_values(['user_id','click_timestamp'])
    histories, labels = {}, {}
    for u, g in clicks.groupby('user_id'):
        rows = list(zip(g.click_article_id.astype(int), g.click_timestamp.astype(int)))
        histories[int(u)], labels[int(u)] = rows[:-1], rows[-1][0]
    train_users, valid_users = train_test_split(users, test_size=.2, random_state=args.seed)
    train_hist = {int(u):histories[int(u)] for u in train_users}
    valid_hist = {int(u):histories[int(u)] for u in valid_users}
    assert not set(train_hist) & set(valid_hist)
    metadata = pd.read_csv(data/'articles.csv').set_index('article_id')
    print('Loading full article embeddings', flush=True)
    emb = pd.read_csv(data/'articles_emb.csv', dtype=np.float32)
    ids = emb.article_id.to_numpy(dtype=np.int64)
    vectors = np.ascontiguousarray(emb.filter(regex='^emb_').to_numpy())
    faiss.normalize_L2(vectors); del emb
    print('Building five-channel retrieval from training histories only', flush=True)
    retrieval = Retrieval(train_hist, metadata, ids, vectors, args.seed, args.epochs)
    train, x, injection = candidate_table(retrieval, train_hist, labels, True)
    valid, vx, _ = candidate_table(retrieval, valid_hist, labels, False)
    print('Candidates:', len(train), len(valid), 'training positives injected:', injection, flush=True)
    assert x.shape[1] == len(FEATURE_NAMES)
    # User-disjoint ranking folds. Retrieval was fitted on outer training histories,
    # without any held-out final clicks; inner OOF shares this fixed retrieval stage.
    oof = np.zeros((len(train),3)); predictions = np.zeros((len(valid),3))
    arrays = din_arrays(train, histories, retrieval)
    valid_arrays = din_arrays(valid, histories, retrieval)
    y = train.label.to_numpy()
    for fold, (ti, vi) in enumerate(GroupKFold(5).split(x, y, train.user_id)):
        assert not set(train.iloc[ti].user_id) & set(train.iloc[vi].user_id)
        print('Ranking fold',fold+1,'/ 5',flush=True)
        order = ti[np.argsort(train.user_id.to_numpy()[ti], kind='stable')]
        sizes = train.iloc[order].groupby('user_id', sort=False).size().to_numpy()
        config = dict(n_estimators=60, num_leaves=15, learning_rate=.05, verbosity=-1,
                      n_jobs=1, random_state=args.seed+fold)
        ranker = lgb.LGBMRanker(objective='lambdarank', **config)
        ranker.fit(x[order], y[order], group=sizes, feature_name=FEATURE_NAMES)
        classifier = lgb.LGBMClassifier(**config)
        classifier.fit(x[ti],y[ti],feature_name=FEATURE_NAMES)
        scaler = StandardScaler().fit(x[ti])
        sx = np.clip(scaler.transform(x),-10,10).astype(np.float32)
        svx = np.clip(scaler.transform(vx),-10,10).astype(np.float32)
        torch.manual_seed(args.seed+fold)
        model = DIN(vectors.shape[1],x.shape[1]); optimizer = torch.optim.Adam(model.parameters(),lr=.001)
        # All positives and up to 20 negatives per positive for neural training.
        rng = np.random.default_rng(args.seed+fold)
        positives = ti[y[ti]==1]; negatives = ti[y[ti]==0]
        if not len(positives):
            raise ValueError('No recalled positive in this fold; increase --users.')
        chosen = np.concatenate([positives,rng.choice(negatives,min(len(negatives),20*len(positives)),replace=False)])
        for epoch in range(args.epochs):
            model.train()
            for batch in np.array_split(rng.permutation(chosen),max(1,math.ceil(len(chosen)/256))):
                logits = model(*(torch.tensor(a[batch]) for a in arrays),torch.tensor(sx[batch]))
                loss = nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor(y[batch],dtype=torch.float32))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        raw_oof = ranker.predict(x[vi]); raw_valid = ranker.predict(vx)
        # Convert ranker scores to per-user percentiles for stable fold blending.
        oof[vi,0] = pd.Series(raw_oof).groupby(train.iloc[vi].user_id.to_numpy()).rank(pct=True).to_numpy()
        predictions[:,0] += pd.Series(raw_valid).groupby(valid.user_id.to_numpy()).rank(pct=True).to_numpy()/5
        oof[vi,1] = classifier.predict_proba(x[vi])[:,1]
        predictions[:,1] += classifier.predict_proba(vx)[:,1]/5
        oof[vi,2] = din_predict(model,tuple(a[vi] for a in arrays),sx[vi])
        predictions[:,2] += din_predict(model,valid_arrays,svx)/5
        torch.save(model.state_dict(),out/f'din_fold_{fold}.pt')
        ranker.booster_.save_model(str(out/f'ranker_fold_{fold}.txt'))
        classifier.booster_.save_model(str(out/f'classifier_fold_{fold}.txt'))
        with (out/f'scaler_fold_{fold}.pkl').open('wb') as f: pickle.dump(scaler,f)
    # Meta-model sees only OOF predictions; outer validation never fits weights.
    stack = LogisticRegression(max_iter=1000,random_state=args.seed).fit(oof,y)
    scores = {'recall_only':vx[:,15], 'lgb_ranker':predictions[:,0],
              'lgb_classifier':predictions[:,1], 'din':predictions[:,2],
              'blend':predictions.mean(axis=1), 'stacking':stack.predict_proba(predictions)[:,1]}
    metrics = {name:evaluate(valid,s,valid_users) for name,s in scores.items()}
    report = {'seed':args.seed,'sample_users':len(users),'train_users':len(train_users),
              'validation_users':len(valid_users),'ranking_folds':5,'features':x.shape[1],
              'training_positives_injected':injection,'epochs':args.epochs,
              'elapsed_seconds':round(time.time()-started,2),'metrics':metrics,
              'limitations':['Sample experiment; historical reported metrics not reproduced.',
                             'YouTubeDNN-style pooled two-tower; does not explicitly encode temporal order.',
                             'FAISS IndexFlatIP is exact search, not ANN.',
                             'Ranking OOF uses fixed outer-training-history retrieval, not nested retrieval refitting.',
                             'Repeated held-out articles already in history are excluded by unseen-item filtering.',
                             'Full article catalogue is not filtered by request time; not a strict temporal causality benchmark.'],
              'training_candidate_hit_rate':float(train.groupby('user_id').label.max().mean()),
              'blend_uplift_pp':100*(metrics['blend']['HitRate@5']-metrics['recall_only']['HitRate@5']),
              'stacking_uplift_pp':100*(metrics['stacking']['HitRate@5']-metrics['recall_only']['HitRate@5'])}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    (out/'feature_names.json').write_text(json.dumps(FEATURE_NAMES,indent=2))
    with (out/'stacking.pkl').open('wb') as f: pickle.dump(stack,f)
    torch.save(retrieval.tower.state_dict(),out/'youtube_dnn.pt')
    valid.assign(**{name:s for name,s in scores.items()}).to_csv(out/'validation_predictions.csv',index=False)
    train.assign(rank_oof=oof[:,0],classifier_oof=oof[:,1],din_oof=oof[:,2]).to_csv(out/'oof_predictions.csv',index=False)
    top = valid.assign(score=scores['stacking']).sort_values(['user_id','score'],ascending=[True,False]).groupby('user_id').head(5)
    assert top.groupby('user_id').size().eq(5).all() and top.user_id.nunique()==len(valid_users)
    assert all(a not in {b for b,_ in histories[u]} for u,a in zip(top.user_id,top.article_id))
    top['rank'] = top.groupby('user_id').cumcount()+1
    top.pivot(index='user_id',columns='rank',values='article_id').rename(columns=lambda c:f'article_{c}').to_csv(out/'top5_validation.csv')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--users',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=2)
    args = parser.parse_args()
    if args.users < 50 or args.epochs < 1:
        parser.error('Use at least 50 users and one epoch.')
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        run(args)
    finally:
        faulthandler.cancel_dump_traceback_later()
