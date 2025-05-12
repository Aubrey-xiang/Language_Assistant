import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1. 读取数据
df_train = pd.read_csv(
    'oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv',
    sep='\t', names=['q1', 'q2', 'label']
).fillna('')
df_test = pd.read_csv(
    'oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv',
    sep='\t', names=['q1', 'q2']
).fillna('')

# 构造文本特征
X = (df_train['q1'] + ' ' + df_train['q2']).values
y = df_train['label'].values
X_test = (df_test['q1'] + ' ' + df_test['q2']).values

# 2. 配置 TF-IDF 向量器
# 使用字符级 1-3 gram，限制特征数加快速度
tfidf_params = {
    'analyzer': 'char',
    'ngram_range': (1, 3),
    'max_features': 50000
}

# 3. 5 折交叉验证与训练
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), start=1):
    print(f"--- Fold {fold} ---")
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    # 每折分别 fit/transform 避免数据泄露
    vectorizer = TfidfVectorizer(**tfidf_params)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)
    X_test_tfidf = vectorizer.transform(X_test)

    # 朴素贝叶斯分类器，适合高维稀疏数据
    clf = ComplementNB()
    clf.fit(X_train_tfidf, y_train)

    # 验证集预测
    oof_preds[valid_idx] = clf.predict_proba(X_valid_tfidf)[:, 1]
    fold_auc = roc_auc_score(y_valid, oof_preds[valid_idx])
    print(f"Fold {fold} AUC: {fold_auc:.4f}")

    # 测试集预测累加
    test_preds += clf.predict_proba(X_test_tfidf)[:, 1] / kf.n_splits

# 计算整体 CV AUC
overall_auc = roc_auc_score(y, oof_preds)
print(f"Overall CV AUC: {overall_auc:.4f}")

# 4. 保存结果
os.makedirs('prediction_result', exist_ok=True)
pd.DataFrame(test_preds, columns=['prob']) \
  .to_csv('prediction_result/simple_nb_result.csv', index=False, header=False)

print("Done.")
