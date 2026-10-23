"""Generate a reviewable portfolio report from saved experiment evidence."""
import argparse
import json
import hashlib
import platform
from importlib.metadata import version
from pathlib import Path
import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',default='temporal_10000_v1')
    parser.add_argument('--report',default='temporal-study.md')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    directory=root/'outputs'/args.study
    config=json.loads((directory/'config.json').read_text())
    reports=[json.loads((directory/f'seed_{seed}.json').read_text()) for seed in config['seeds']]
    evidence={'python':platform.python_version(),'platform':platform.platform(),
              'packages':{name:version(name) for name in ['numpy','pandas','scikit-learn','faiss-cpu','lightgbm','torch']},
              'sha256':{}}
    for path in [root/'src'/'study.py',root/'src'/'pipeline.py',*sorted((root/'data').glob('*.csv'))]:
        digest=hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda:source.read(8*1024*1024),b''):digest.update(block)
        evidence['sha256'][str(path.relative_to(root))]=digest.hexdigest()
    (directory/'evidence_manifest.json').write_text(json.dumps(evidence,indent=2))
    summary=pd.read_csv(directory/'summary.csv')
    lines=['# 新闻推荐系统：时间评估与消融实验报告','',
           '本报告根据本地保存的逐用户划分、候选预测和指标生成，不使用历史面试稿的宣称指标。','',
           '## 评估规则','',
           f"每个随机种子抽取 {config['users']:,} 名至少两次点击的用户，种子为 {config['seeds']}。保留最后一次点击为标签，倒数第二次点击为推荐请求时刻。",
           '按标签时间的 80% 分位切分：训练标签不晚于切分点，验证请求不早于切分点；跨界用户进入单独的 gap 审计，不悄悄混入训练或验证。训练和验证用户互斥。',
           f"召回统计和双塔使用 {config['snapshots']} 个训练历史快照；验证使用切分点处的固定快照。每个快照的所有源点击都不晚于它服务的最早请求，最终点击标签不参与这些统计。",
           '文章候选在请求时刻已发布且不在用户历史中。为避免先截断后过滤，发布时间约束在每路排序截断前应用。内容检索预取 256 个邻居，属于有上限的候选搜索，不保证穷尽所有可用文章。',
           '训练和验证候选都不注入正例。HR/NDCG 使用全部验证用户作为分母，包括重复点击或当时未发布的真实标签；另报可服务用户指标。GAUC 只在存在正负候选的用户上等权计算。',
           '未过滤候选的对照保留同样的因果训练快照和 Ranker，专门检查候选发布时间约束的影响，不重用此前随机用户划分的数值。',
           '所有排序实验采用同一固定配置的 LightGBM Ranker。每个召回/特征消融都重新训练；不利用外层验证调参。','',
           '## 样本、时间与标签审计','',
           '| 种子 | 训练用户 | 验证用户 | 跨界用户 | 重复标签 | 未发布标签 | 可服务用户 | Top-5 不足用户 | 耗时/秒 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        a=r['label_audit']
        lines.append(f"| {r['seed']} | {r['train_users']} | {r['valid_users']} | {r['temporal_gap_users']} | {a['repeated_labels']} | {a['unpublished_labels']} | {a['servable_users']} | {r['short_top5_users']} | {r['elapsed_seconds']} |")
        for snap in r['snapshot_audits']:
            assert snap['snapshot_time']<=snap['earliest_request']
            assert snap['source_max_event_time'] is None or snap['source_max_event_time']<=snap['snapshot_time']
    def value(experiment,metric,percent=False):
        row=summary[(summary.experiment==experiment)&(summary.metric==metric)].iloc[0]
        if pd.isna(row['mean']):return '无有效用户'
        factor=100 if percent else 1
        return f"{row['mean']*factor:.3f} ± {row['std']*factor:.3f}"+('%' if percent else '')
    lines+=['','## 总体结果：三种子均值 ± 样本标准差','',
            '标准差表示这些随机抽样结果的波动，不是置信区间；种子间可能有重叠用户。','',
            '| 实验 | HitRate@5 | NDCG@5 | 候选覆盖率 | 条件 GAUC |',
            '|---|---:|---:|---:|---:|']
    for name in reports[0]['results']:
        lines.append(f"| {name} | {value(name,'HitRate@5',True)} | {value(name,'NDCG@5')} | {value(name,'candidate_hit_rate',True)} | {value(name,'GAUC_recalled_users_equal_weight')} |")
    lines+=['','## Ranker 用户分群','',
            '| 种子 | 分群 | 用户数 | HitRate@5 | NDCG@5 | 候选覆盖率 | GAUC 有效用户 |',
            '|---|---|---:|---:|---:|---:|---:|']
    for r in reports:
        for name,m in r['results']['ranker'].items():
            lines.append(f"| {r['seed']} | {name} | {m['evaluation_users']} | {100*m['HitRate@5']:.3f}% | {m['NDCG@5']:.4f} | {100*m['candidate_hit_rate']:.3f}% | {m['GAUC_eligible_users']} |")
    base=summary[(summary.experiment=='ranker')&(summary.metric=='HitRate@5')].iloc[0]['mean']
    lines+=['','## 消融差值','',
            '下表是移除后减去完整 Ranker 的命中率差，单位百分点；负值代表这次移除降低效果。小幅差异不直接证明统计显著。','',
            '| 消融实验 | HitRate@5 差值/百分点 |','|---|---:|']
    for _,r in summary[(summary.metric=='HitRate@5')&summary.experiment.str.startswith('without_')].iterrows():
        lines.append(f"| {r['experiment']} | {100*(r['mean']-base):+.3f} |")
    lines+=['','## 证据与局限','',
            f"原始证据位于 `outputs/{args.study}/`：配置、每种子 JSON、用户划分 CSV、逐候选预测、Top-5、特征重要性和排序模型。",'',
            '- 验证召回使用训练结束时的固定快照，因此存在新闻热度与话题的时间漂移。低覆盖率可能反映快照陈旧，不能仅归因为排序模型。下一步应添加过去事件驱动的滚动召回更新。',
            '- 扩展训练快照较粗，最早快照可能没有足够数据，未训练的双塔路线被禁用。训练配置一致不等于所有路线充分训练。',
            '- 当前内容向量来自原始预计算文件，其预训练语料和截止时间没有可验证记录，不能声称内容表征完全无未来语料信息。',
            '- 精确 FAISS 检索、池化双塔、单机训练，不宣称已实现 ANN 加速、显式时序建模或线上部署。',
            '- 这次研究重点是时间协议、规模稳定性和消融，不重复调参比较 DIN/Stacking。旧的随机用户实验继续单独保留。',
            '- 旧实验的 21% 与本次时间实验不能直接比较，也不能把全部差异归因于发布时间过滤：用户抽样规模、用户切分、训练快照、训练目标时段等同时变化。隔离发布时间过滤的对照是本报告的 unfiltered_ranker。',
            '- 全量冷启动服务、未来文章发布策略、重复阅读业务规则需要另行定义；不通过删除困难用户美化主指标。','']
    if config.get('rolling_bins',0):
        def mean(name,metric):
            return summary[(summary.experiment==name)&(summary.metric==metric)].iloc[0]['mean']
        fixed=mean('ranker','HitRate@5');rolling=mean('rolling_ranker','HitRate@5')
        lines.insert(2,f"本轮同用户、同时间切分、同排序器对照：HR@5 从 {fixed*100:.3f}% 到 {rolling*100:.3f}%（{(rolling-fixed)*100:+.3f} 个百分点）；候选覆盖率从 {mean('ranker','candidate_hit_rate')*100:.3f}% 到 {mean('rolling_ranker','candidate_hit_rate')*100:.3f}%。三个种子方向一致不等于统计显著；绝对覆盖率仍低，不宣称生产推荐质量。")
        lines[lines.index('- 验证召回使用训练结束时的固定快照，因此存在新闻热度与话题的时间漂移。低覆盖率可能反映快照陈旧，不能仅归因为排序模型。下一步应添加过去事件驱动的滚动召回更新。')]='- 本报告同时对照固定快照与滚动刷新。滚动来源仅为训练用户：在各验证时间段最早请求之前已观察到的历史及最终点击。验证用户及其标签不进入召回来源；排序器不重新训练。'
        lines.insert(4,f"滚动实验按验证请求时间分为 {config['rolling_bins']} 段，每段重建全部召回及其统计特征；双塔也以相同配置重训，因此不是仅更新热门榜的单因素实验。固定基线仍不使用最终训练点击，而滚动实验加入当时已知的训练最终点击，这是可观测信息更新的一部分。")
        for seed in config['seeds']:
            audits=json.loads((directory/f'rolling_audit_seed_{seed}.json').read_text())
            assert all(a['source_max_event_time']<=a['snapshot_time']<=a['earliest_request'] for a in audits)
    target=root/'docs'/args.report
    target.write_text('\n'.join(lines))
    print(target)


if __name__=='__main__':main()
