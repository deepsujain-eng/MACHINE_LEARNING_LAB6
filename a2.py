import unittest
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder

def clean_encode_data(df):
    df_enc = df.copy()
    for col in df_enc.columns:
        if df_enc[col].dtype == 'object':
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    return df_enc

def fill_missing_data(df, strategy='mean'):
    df_imp = df.copy()
    for col in df_imp.columns:
        if df_imp[col].isnull().any():
            if strategy == 'mean':
                val = df_imp[col].mean()
            elif strategy == 'median':
                val = df_imp[col].median()
            elif strategy == 'mode':
                val = df_imp[col].mode()[0]
            else:
                val = df_imp[col].mean()
            df_imp[col] = df_imp[col].fillna(val)
    return df_imp

def calc_euclidean(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def sort_dist_insertion(dist_list):
    d = dist_list.copy()
    for i in range(1, len(d)):
        key = d[i]
        j = i - 1
        while j >= 0 and d[j][0] > key[0]:
            d[j + 1] = d[j]
            j -= 1
        d[j + 1] = key
    return d

def get_k_neighbors(dist_list, k):
    return dist_list[:k]

def predict_majority(neighbors):
    counts = Counter()
    for d, label in neighbors:
        counts[label] += 1
    max_count = max(counts.values())
    winners = [lbl for lbl, cnt in counts.items() if cnt == max_count]
    return sorted(winners, key=lambda x: str(x))[0]

class TestKNNFunctions(unittest.TestCase):

    def test_encode_data(self):
        df = pd.DataFrame({'col1': ['apple', 'banana', 'apple']})
        encoded = clean_encode_data(df)
        self.assertTrue(pd.api.types.is_numeric_dtype(encoded['col1']))

    def test_fill_missing_data(self):
        df = pd.DataFrame({'col1': [1, 2, np.nan, 4]})
        imputed = fill_missing_data(df, 'mean')
        self.assertFalse(imputed['col1'].isnull().any())
        self.assertEqual(imputed['col1'].iloc[2], 2.3333333333333335)

    def test_calc_euclidean(self):
        p1 = np.array([0, 0])
        p2 = np.array([3, 4])
        dist = calc_euclidean(p1, p2)
        self.assertEqual(dist, 5.0)

    def test_sort_dist_insertion(self):
        dists = [(5.0, 1), (1.0, 0), (3.0, 0)]
        sorted_d = sort_dist_insertion(dists)
        self.assertEqual(sorted_d[0][0], 1.0)
        self.assertEqual(sorted_d[1][0], 3.0)
        self.assertEqual(sorted_d[2][0], 5.0)

    def test_get_k_neighbors(self):
        dists = [(1.0, 0), (2.0, 1), (3.0, 0)]
        neighbors = get_k_neighbors(dists, 2)
        self.assertEqual(len(neighbors), 2)

    def test_predict_majority(self):
        neighbors = [(1.0, 1), (1.5, 1), (2.0, 0)]
        pred = predict_majority(neighbors)
        self.assertEqual(pred, 1)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)