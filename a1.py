import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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

class GenAIKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=float)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        preds = []
        for pt in X:
            dists = []
            for idx, train_pt in enumerate(self.X_train):
                d = calc_euclidean(pt, train_pt)
                dists.append((d, self.y_train[idx]))
            sorted_dists = sort_dist_insertion(dists)
            neighbors = get_k_neighbors(sorted_dists, self.k)
            pred = predict_majority(neighbors)
            preds.append(pred)
        return np.array(preds)

df = pd.read_excel('Parkinsons_Perturbation_MFCC_Features.xlsx', sheet_name=0)
if 'id' in df.columns:
    df = df.drop(columns=['id'])

df = fill_missing_data(df, 'mean')
X = clean_encode_data(df.drop('class', axis=1)).values
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GenAIKNN(k=3)
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(accuracy_score(y_test, preds))