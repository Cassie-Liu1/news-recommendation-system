import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('pipeline', Path(__file__).resolve().parents[1]/'src'/'pipeline.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class EvaluationTests(unittest.TestCase):
    def test_unrecalled_users_stay_in_denominator(self):
        table = pd.DataFrame([(1, 10, 1), (1, 11, 0), (2, 12, 0)],
                             columns=['user_id', 'article_id', 'label'])
        result = p.evaluate(table, np.array([.9, .1, .3]), [1, 2, 3])
        self.assertAlmostEqual(result['HitRate@5'], 1/3)
        self.assertAlmostEqual(result['NDCG@5'], 1/3)
        self.assertEqual(result['GAUC_eligible_users'], 1)
        self.assertEqual(result['GAUC_recalled_users_equal_weight'], 1)

    def test_position_discount_and_ties(self):
        table = pd.DataFrame([(1, 10, 0), (1, 11, 1)], columns=['user_id', 'article_id', 'label'])
        result = p.evaluate(table, np.array([.5, .5]), [1])
        self.assertAlmostEqual(result['NDCG@5'], 1/np.log2(3))
        self.assertAlmostEqual(result['GAUC_recalled_users_equal_weight'], .5)

    def test_labels_cannot_inject_candidates(self):
        class FakeRetrieval:
            def retrieve(self, rows): return {10: {'popular': (1, 2.)}}
            def features(self, rows, article, sources): return [1., 2.]
        for training in [False, True]:
            table, _, injected = p.candidate_table(FakeRetrieval(), {1: [(8, 0)]}, {1: 99}, training)
            self.assertEqual(table.article_id.tolist(), [10])
            self.assertEqual(table.label.sum(), 0)
            self.assertEqual(injected, 0)

    def test_single_history_feature_units(self):
        r = p.Retrieval.__new__(p.Retrieval)
        r.article_info = {8: (2, 86400000., 100), 10: (2, 0., 150)}
        r.history_features = {}
        r.position = {8: 0, 10: 1}
        r.vectors = np.array([[1., 0.], [1., 0.]], dtype=np.float32)
        from collections import Counter
        r.popularity = Counter({10: 3})
        r.unique_users = {10: 2}
        values = r.features([(8, 172800000)], 10, {'popular': (1, 3.)})
        self.assertEqual(len(values), len(p.FEATURE_NAMES))
        f = dict(zip(p.FEATURE_NAMES, values))
        self.assertEqual(f['article_age_days'], 2.)
        self.assertEqual(f['creation_gap_days'], 1.)
        self.assertEqual(f['mean_interclick_seconds'], 0.)
        self.assertEqual(f['word_preference_gap'], 50.)
        self.assertEqual(f['itemcf_present'], 0.)
        self.assertEqual(f['popular_present'], 1.)

    def test_sparse_history_never_recommends_seen_article(self):
        p.torch.set_num_threads(1)
        p.faiss.omp_set_num_threads(1)
        ids = np.arange(1, 7, dtype=np.int64)
        metadata = pd.DataFrame({'article_id': ids, 'category_id': [1]*6,
                                 'created_at_ts': [0]*6, 'words_count': [100]*6}).set_index('article_id')
        retrieval = p.Retrieval({1: [(1, 0)]}, metadata, ids, np.eye(6, dtype=np.float32), 42, 1)
        candidates = retrieval.retrieve([(1, 0)])
        self.assertNotIn(1, candidates)
        self.assertEqual(len(candidates), 5)
        self.assertTrue(all(a in set(ids) for a in candidates))


if __name__ == '__main__':
    unittest.main()
