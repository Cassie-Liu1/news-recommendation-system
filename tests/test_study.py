import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import study as s


class CausalityTests(unittest.TestCase):
    def test_rolling_source_only_observed_training_events(self):
        past=s.rolling_source({1:[(10,50)],2:[(10,70)]},
                             {1:11,2:12},{1:90,2:120},{10:0,11:80,12:0},100)
        self.assertEqual(past,{1:[(10,50),(11,90)],2:[(10,70)]})

    def test_temporal_split_has_no_future_training_label(self):
        histories={u:[(1,u*10)] for u in range(1,11)}
        labels={u:u*10+1 for u in histories}
        train,valid,gap,cutoff=s.split_users(histories,labels,42)
        self.assertTrue(set(train).isdisjoint(valid))
        self.assertLessEqual(max(labels[u] for u in train),min(histories[u][-1][1] for u in valid))
        self.assertEqual(len(train)+len(valid)+len(gap),10)

    def test_future_source_event_is_rejected(self):
        with self.assertRaises(AssertionError):s.assert_snapshot({1:[(10,101)]},100)

    def test_ablation_removes_unique_candidates_and_recomputes_rrf(self):
        table=s.pd.DataFrame([(1,10,1),(1,11,0)],columns=['user_id','article_id','label'])
        x=s.np.zeros((2,len(s.FEATURE_NAMES)),dtype=s.np.float32)
        x[0,16:19]=[1,1,4]
        x[1,16:19]=[1,1,4];x[1,19:22]=[1,2,3]
        result,features=s.channel_ablation(table,x,'itemcf')
        self.assertEqual(result.article_id.tolist(),[11])
        self.assertEqual(features[0,14],1)
        self.assertAlmostEqual(features[0,15],1/62)
        self.assertEqual(features[0,16],0)

    def test_future_and_repeat_labels_are_audited_not_silently_dropped(self):
        metadata=s.pd.DataFrame({'article_id':[1,2,3],'created_at_ts':[0,200,0]}).set_index('article_id')
        histories={1:[(1,100)],2:[(1,100)],3:[(1,100)]}
        audit,servable=s.label_audit([1,2,3],histories,{1:1,2:2,3:3},metadata)
        self.assertEqual(audit['users'],3)
        self.assertEqual(audit['repeated_labels'],1)
        self.assertEqual(audit['unpublished_labels'],1)
        self.assertEqual(servable,[3])

    def test_actual_retrieval_filters_future_articles(self):
        s.torch.set_num_threads(1);s.faiss.omp_set_num_threads(1)
        ids=s.np.arange(1,7,dtype=s.np.int64)
        metadata=s.pd.DataFrame({'article_id':ids,'category_id':[1]*6,
                                 'created_at_ts':[0,50,150,200,250,300],
                                 'words_count':[100]*6}).set_index('article_id')
        retriever=s.CausalRetrieval({9:[(1,80)]},metadata,ids,s.np.eye(6,dtype=s.np.float32),42,1,80)
        strict=retriever.candidates([(1,100)],10,30,True)
        self.assertEqual(set(strict),{2})
        self.assertNotIn(1,strict)
        for sources in strict.values():self.assertNotIn('youtube_dnn',sources)


if __name__=='__main__':unittest.main()
