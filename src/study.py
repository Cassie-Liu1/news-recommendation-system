"""Reproducible temporal/user-disjoint study, multi-seed and retrained ablations.

Expanding retrieval snapshots never include an event later than any request
served by that snapshot. Ranking training targets precede validation requests.
The catalogue itself is filtered at each request (not at the label timestamp).
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import time

from pipeline import (CHANNELS, FEATURE_NAMES, Retrieval, evaluate, faiss, lgb,
                      np, pd, torch)


def split_users(histories, label_times, seed):
    """Chronological target split; boundary-crossing histories form an audited gap."""
    cutoff = int(np.quantile(list(label_times.values()), .8))
    train = [u for u in histories if label_times[u] <= cutoff]
    valid = [u for u in histories if histories[u][-1][1] >= cutoff and label_times[u] > cutoff]
    train_set, valid_set = set(train), set(valid)
    gap = [u for u in histories if u not in train_set and u not in valid_set]
    assert set(train).isdisjoint(valid)
    assert max(label_times[u] for u in train) <= min(histories[u][-1][1] for u in valid)
    return train, valid, gap, cutoff


def label_audit(users, histories, labels, metadata):
    repeats = [u for u in users if labels[u] in {a for a, _ in histories[u]}]
    future = [u for u in users if int(metadata.loc[labels[u], 'created_at_ts']) > histories[u][-1][1]]
    available = [u for u in users if u not in set(repeats) | set(future)]
    return {'users': len(users), 'repeated_labels': len(repeats), 'unpublished_labels': len(future),
            'repeat_and_unpublished': len(set(repeats) & set(future)), 'servable_users': len(available)}, available


def assert_snapshot(histories, cutoff):
    assert all(t <= cutoff for rows in histories.values() for _, t in rows)


def rolling_source(train_hist, labels, times, published, boundary):
    """Observed training-user events only; validation users/targets never enter."""
    past = {}
    for u, rows in train_hist.items():
        events = rows + [(labels[u], times[u])]
        eligible = [(a,t) for a,t in events if t <= boundary and int(published[a]) <= t]
        if eligible:
            past[u] = eligible
    assert_snapshot(past, boundary)
    return past


def channel_ablation(table, x, removed=None, popular_only=False):
    """Remove candidates unique to a removed route, then recompute union features."""
    active = ['popular'] if popular_only else [c for c in CHANNELS if c != removed]
    keep = np.zeros(len(table), dtype=bool)
    result = x.copy()
    for c in CHANNELS:
        start = 16 + CHANNELS.index(c)*3
        if c in active:
            keep |= result[:, start] > 0
        else:
            result[:, start:start+3] = 0
    result[:, 14] = sum(result[:,16+CHANNELS.index(c)*3] for c in active)
    result[:, 15] = sum(np.where(result[:,16+CHANNELS.index(c)*3] > 0,
                                1/(60+result[:,17+CHANNELS.index(c)*3]), 0) for c in active)
    return table.loc[keep].reset_index(drop=True), result[keep]


def feature_columns(removed):
    groups = {'content': ['last_content_similarity', 'max_content_similarity', 'mean_content_similarity',
                           'content_present', 'content_rank', 'content_score'],
              'time': ['history_span_days', 'mean_interclick_seconds', 'article_age_days', 'creation_gap_days'],
              'source': FEATURE_NAMES[14:]}
    return [i for i, name in enumerate(FEATURE_NAMES) if name not in groups.get(removed, [])]


class CausalRetrieval(Retrieval):
    def __init__(self, histories, metadata, ids, vectors, seed, epochs, cutoff):
        assert_snapshot(histories, cutoff)
        self.snapshot_time = cutoff
        negative_positions=np.flatnonzero(metadata.loc[ids,'created_at_ts'].to_numpy()<=cutoff)
        super().__init__(histories, metadata, ids, vectors, seed, epochs,negative_positions)
        self.training_pairs = sum(max(0, min(len(rows), 20)-1) for rows in histories.values())

    def warm(self, histories, count):
        unique = sorted({a for rows in histories.values() for a,_ in rows[-10:]})
        # Batched BLAS is much faster than many independent single-query searches.
        for start in range(0, len(unique), 128):
            articles = unique[start:start+128]
            queries = self.vectors[[self.position[a] for a in articles]]
            scores, positions = self.content.search(np.ascontiguousarray(queries), max(count+10,256))
            for a, ss, pp in zip(articles, scores, positions):
                self.cache[a] = [(int(self.ids[b]),float(s)) for b,s in zip(pp,ss) if b>=0]

    def candidates(self, rows, user_id, count, strict=True):
        request = rows[-1][1]
        assert self.snapshot_time <= request
        # Keep extra neighbors before filtering; refetching is not an ANN claim.
        candidates = super().retrieve(rows, count=count, user_id=user_id,as_of=request if strict else None)
        if not self.training_pairs:
            candidates = {a:{c:v for c,v in source.items() if c != 'youtube_dnn'} for a,source in candidates.items()}
        if strict:
            candidates = {a:s for a,s in candidates.items() if self.article_info[a][1] <= request and s}
        return candidates


def table_for(retrieval, histories, labels, count, strict=True):
    records, features = [], []
    for number, (u, rows) in enumerate(histories.items(), 1):
        for article, sources in retrieval.candidates(rows,u,count,strict).items():
            records.append((u,article,int(article==labels[u])))
            features.append(retrieval.features(rows,article,sources))
        if number%1000==0:
            print('  candidate requests',number,'/',len(histories),flush=True)
    table = pd.DataFrame(records,columns=['user_id','article_id','label'])
    return table, np.asarray(features,dtype=np.float32).reshape(-1,len(FEATURE_NAMES))


def fit_ranker(train, x, valid, vx, seed, columns):
    order = np.argsort(train.user_id.to_numpy(),kind='stable')
    groups = train.iloc[order].groupby('user_id',sort=False).size().to_numpy()
    if train.label.nunique()!=2:
        raise ValueError('No recalled positives in temporal training candidates.')
    model = lgb.LGBMRanker(objective='lambdarank',n_estimators=80,num_leaves=15,
                          learning_rate=.05,n_jobs=1,verbosity=-1,random_state=seed)
    model.fit(x[order][:,columns],train.label.to_numpy()[order],group=groups,
              feature_name=[FEATURE_NAMES[i] for i in columns])
    return model, model.booster_.predict(vx[:,columns])


def grouped_metrics(table,scores,users,histories,servable):
    groups = {'all': users, 'servable':servable,
              'one_history_click':[u for u in users if len(histories[u])==1],
              'two_to_five_history_clicks':[u for u in users if 2<=len(histories[u])<=5],
              'six_plus_history_clicks':[u for u in users if len(histories[u])>=6]}
    result = {}
    for name, members in groups.items():
        if members:
            mask = table.user_id.isin(members).to_numpy()
            result[name] = evaluate(table.loc[mask],scores[mask],members)
    return result


def run_seed(args, seed, all_histories, labels_all, times_all, metadata, ids, vectors, out):
    started = time.time()
    users = np.random.default_rng(seed).choice(sorted(all_histories),args.users,replace=False)
    histories = {int(u):all_histories[int(u)] for u in sorted(users)}
    labels = {u:labels_all[u] for u in histories}; times={u:times_all[u] for u in histories}
    train_users,valid_users,gap,cutoff=split_users(histories,times,seed)
    audit,servable=label_audit(valid_users,histories,labels,metadata)
    train_hist={u:histories[u] for u in train_users}; valid_hist={u:histories[u] for u in valid_users}
    if not train_users or not valid_users:
        raise ValueError('Temporal split has no training or validation users.')
    # Expanding snapshots at the earliest request of each training time bin.
    # Only histories of outer training users are statistic/model sources.
    ordered=sorted(train_users,key=lambda u:histories[u][-1][1])
    bins=[list(map(int,b)) for b in np.array_split(ordered,args.snapshots) if len(b)]
    bins.append(valid_users)
    train_parts=[]; train_x=[]; valid= vx= unfiltered= ux=None; snapshot_audits=[]
    published=metadata.created_at_ts.to_dict()
    for number, members in enumerate(bins):
        boundary=min(histories[u][-1][1] for u in members)
        if number==len(bins)-1:
            boundary=cutoff
        past={u:[(a,t) for a,t in rows if t<=boundary and int(published[a])<=t]
              for u,rows in train_hist.items()}
        past={u:rows for u,rows in past.items() if rows}
        print('Seed',seed,'snapshot',number+1,'/',len(bins),'source users',len(past),flush=True)
        retrieval=CausalRetrieval(past,metadata,ids,vectors,seed,args.epochs,boundary)
        queries={u:histories[u] for u in members}
        retrieval.warm(queries,args.candidates)
        table,features=table_for(retrieval,queries,labels,args.candidates,True)
        assert all(int(published[a]) <= histories[u][-1][1]
                   for u,a in zip(table.user_id,table.article_id))
        snapshot_audits.append({'snapshot_time':boundary,'earliest_request':min(histories[u][-1][1] for u in members),
                                'source_users':len(past),'events':sum(map(len,past.values())),
                                'source_max_event_time':max((t for rows in past.values() for _,t in rows),default=None),
                                'training_pairs':retrieval.training_pairs})
        if number==len(bins)-1:
            valid,vx=table,features
            unfiltered,ux=table_for(retrieval,queries,labels,args.candidates,False)
        else:
            train_parts.append(table); train_x.append(features)
        del retrieval; gc.collect()
    train=pd.concat(train_parts,ignore_index=True);x=np.concatenate(train_x)
    assert set(train.user_id).isdisjoint(valid.user_id)
    result={}; rows=[]
    def save_result(name,table,score):
        result[name]=grouped_metrics(table,score,valid_users,histories,servable)
        print(name,json.dumps(result[name]['all']),flush=True)
        return table.assign(score=score)
    popular,popx=channel_ablation(valid,vx,popular_only=True)
    save_result('popular_only',popular,popx[:,FEATURE_NAMES.index('popular_score')])
    save_result('recall_rrf',valid,vx[:,15])
    save_result('unfiltered_recall_rrf',unfiltered,ux[:,15])
    model,scores=fit_ranker(train,x,valid,vx,seed,list(range(len(FEATURE_NAMES))))
    full=save_result('ranker',valid,scores)
    model.booster_.save_model(str(out/f'ranker_seed_{seed}.txt'))
    pd.DataFrame({'feature':FEATURE_NAMES,'gain':model.booster_.feature_importance('gain')}).to_csv(out/f'feature_importance_seed_{seed}.csv',index=False)
    full.to_csv(out/f'validation_predictions_seed_{seed}.csv',index=False)
    # Same causal training snapshot/labels: this comparison isolates candidate
    # availability filtering, not a claim that the old noncausal run was valid.
    save_result('unfiltered_ranker',unfiltered,model.booster_.predict(ux))
    if args.rolling_bins:
        rolling_parts=[]; rolling_features=[]; rolling_audits=[]
        ordered_valid=sorted(valid_users,key=lambda u:histories[u][-1][1])
        for number, members in enumerate(np.array_split(ordered_valid,args.rolling_bins)):
            if not len(members): continue
            queries={int(u):histories[int(u)] for u in members}
            boundary=min(rows[-1][1] for rows in queries.values())
            past=rolling_source(train_hist,labels,times,published,boundary)
            print('Rolling seed',seed,'bin',number+1,'/',args.rolling_bins,flush=True)
            retriever=CausalRetrieval(past,metadata,ids,vectors,seed,args.epochs,boundary)
            retriever.warm(queries,args.candidates)
            rt,rx=table_for(retriever,queries,labels,args.candidates)
            assert all(int(published[a]) <= histories[u][-1][1] and
                       a not in {b for b,_ in histories[u]} for u,a in zip(rt.user_id,rt.article_id))
            rolling_parts.append(rt);rolling_features.append(rx)
            rolling_audits.append({'snapshot_time':boundary,'earliest_request':boundary,
                'source_max_event_time':max(t for rows in past.values() for _,t in rows),
                'source_users':len(past),'events':sum(map(len,past.values()))})
            del retriever;gc.collect()
        rt=pd.concat(rolling_parts,ignore_index=True);rx=np.concatenate(rolling_features)
        save_result('rolling_recall_rrf',rt,rx[:,15])
        rf=save_result('rolling_ranker',rt,model.booster_.predict(rx))
        rf.to_csv(out/f'rolling_predictions_seed_{seed}.csv',index=False)
        (out/f'rolling_audit_seed_{seed}.json').write_text(json.dumps(rolling_audits,indent=2))
    for channel in CHANNELS:
        print('Retraining without channel',channel,flush=True)
        tt,tx=channel_ablation(train,x,removed=channel);vt,vtx=channel_ablation(valid,vx,removed=channel)
        _,score=fit_ranker(tt,tx,vt,vtx,seed,list(range(len(FEATURE_NAMES))))
        save_result('without_'+channel,vt,score)
    for group in ['content','time','source']:
        print('Retraining without features',group,flush=True)
        _,score=fit_ranker(train,x,valid,vx,seed,feature_columns(group))
        save_result('without_'+group+'_features',valid,score)
    top=full.sort_values(['user_id','score','article_id'],ascending=[True,False,True]).groupby('user_id').head(5)
    assert all(a not in {b for b,_ in histories[u]} for u,a in zip(top.user_id,top.article_id))
    assert top.groupby('user_id').article_id.nunique().le(5).all()
    top['rank']=top.groupby('user_id').cumcount()+1
    top.pivot(index='user_id',columns='rank',values='article_id').rename(columns=lambda c:f'article_{c}').to_csv(out/f'top5_seed_{seed}.csv')
    short_users=len(valid_users)-int(top.groupby('user_id').size().eq(5).sum())
    report={'seed':seed,'sample_users':args.users,'train_users':len(train_users),'valid_users':len(valid_users),
            'temporal_gap_users':len(gap),'cutoff':cutoff,'label_audit':audit,'snapshot_audits':snapshot_audits,
            'results':result,'short_top5_users':short_users,'training_candidates':len(train),'validation_candidates':len(valid),
            'elapsed_seconds':round(time.time()-started,2)}
    (out/f'seed_{seed}.json').write_text(json.dumps(report,indent=2))
    manifest=[{'user_id':u,'split':'train' if u in set(train_users) else 'valid' if u in set(valid_users) else 'gap',
               'request_time':histories[u][-1][1],'label_time':times[u],'label_article':labels[u]} for u in histories]
    pd.DataFrame(manifest).to_csv(out/f'split_seed_{seed}.csv',index=False)
    return report


def aggregate(reports):
    rows=[]
    for name in reports[0]['results']:
        for metric in ['HitRate@5','NDCG@5','candidate_hit_rate','GAUC_recalled_users_equal_weight']:
            values=[r['results'][name]['all'][metric] for r in reports]
            values=[v for v in values if v is not None]
            rows.append({'experiment':name,'metric':metric,'mean':float(np.mean(values)) if values else None,
                         'std':float(np.std(values,ddof=1)) if len(values)>1 else 0. if values else None,'seeds':len(values)})
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--users',type=int,default=10000)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,123,2026])
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--snapshots',type=int,default=4)
    parser.add_argument('--candidates',type=int,default=30)
    parser.add_argument('--output',default='temporal_study')
    parser.add_argument('--rolling-bins',type=int,default=0,
                        help='Refresh validation retrieval in this many chronological bins; fixed ranker.')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];out=root/'outputs'/args.output;out.mkdir(parents=True,exist_ok=True)
    faiss.omp_set_num_threads(1);torch.set_num_threads(1)
    data=root/'data';clicks=pd.read_csv(data/'train_click_log.csv').drop_duplicates(['user_id','click_article_id','click_timestamp'])
    test_users=pd.read_csv(data/'testA_click_log.csv',usecols=['user_id']).user_id.unique()
    assert set(clicks.user_id).isdisjoint(test_users)
    histories={};labels={};times={}
    for u,g in clicks.sort_values(['user_id','click_timestamp']).groupby('user_id'):
        if len(g)<2:continue
        records=list(zip(g.click_article_id.astype(int),g.click_timestamp.astype(int)))
        histories[int(u)]=records[:-1];labels[int(u)]=records[-1][0];times[int(u)]=records[-1][1]
    if args.users>len(histories) or args.users<100 or args.snapshots<1:
        parser.error('Invalid sample size or snapshots.')
    metadata=pd.read_csv(data/'articles.csv').set_index('article_id')
    emb=pd.read_csv(data/'articles_emb.csv',dtype=np.float32);ids=emb.article_id.to_numpy(dtype=np.int64)
    vectors=np.ascontiguousarray(emb.filter(regex='^emb_').to_numpy());faiss.normalize_L2(vectors);del emb
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    reports=[]
    for seed in args.seeds:
        path=out/f'seed_{seed}.json'
        if path.exists():
            raise FileExistsError(f'{path}: choose a new --output to preserve prior evidence.')
        reports.append(run_seed(args,seed,histories,labels,times,metadata,ids,vectors,out))
    aggregate(reports).to_csv(out/'summary.csv',index=False)
    print('Completed',out,flush=True)


if __name__=='__main__':main()
