"""Reproducible SYNTHETIC beauty review classification demonstration.
Run from repository root: python src/run_project.py
No scraped, private, or real customer records are included.
"""
from pathlib import Path
import json
import platform
import re
import hashlib
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (precision_score, recall_score, f1_score,
    accuracy_score, average_precision_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, ConfusionMatrixDisplay)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
DATA = ROOT / 'data'
SEED = 42

def make_data():
    rng = np.random.default_rng(SEED)
    rows = []
    good = ['absorbs well', 'feels comfortable', 'pleasant texture', 'good value',
            'easy to apply', 'packaging works well', 'met my expectations']
    bad = ['feels sticky', 'poor value', 'strong fragrance', 'bottle leaked',
           'not as described', 'difficult to apply', 'disappointing texture']
    neutral = ['used for a week', 'ordered online', 'arrived yesterday',
               'tried after cleansing', 'bought for daily use', 'small bottle']
    for i in range(1800):
        product = i % 120
        category = ['moisturizer', 'cleanser', 'shampoo', 'body lotion'][product % 4]
        negative = rng.random() < 0.30
        # Deliberately imperfect rating proxy and overlapping language.
        phrases = rng.choice(bad if negative else good, 2, replace=False).tolist()
        if rng.random() < .35:
            phrases[1] = rng.choice(good if negative else bad)
        if rng.random() < .12:
            negative = not negative
        rating = int(rng.choice([1, 2] if negative else [3, 4, 5]))
        text = f'This {category}: {phrases[0]}, {phrases[1]}. {rng.choice(neutral)}.'
        rows.append(dict(review_id=f'SYN{i:05}', product_id=f'P{product:03}',
                         category=category, date=(pd.Timestamp('2025-01-01')+
                         pd.Timedelta(days=int(rng.integers(0,365)))).isoformat(),
                         rating=rating, text=text, verified=bool(rng.random()<.8),
                         provenance='synthetic'))
    df = pd.DataFrame(rows)
    # Controlled errors make the cleaning audit testable.
    df.loc[0:9, 'text'] = None
    df.loc[10:14, 'rating'] = 7
    df.loc[15:19, 'rating'] = None
    df.loc[20:24, 'date'] = 'not-a-date'
    df = pd.concat([df, df.iloc[30:50]], ignore_index=True)
    return df

def clean(raw):
    d = raw.copy()
    audit = {'raw_rows': len(d)}
    audit['duplicate_ids_removed'] = int(d.review_id.duplicated().sum())
    d = d.drop_duplicates('review_id', keep='first').copy()
    d['rating'] = pd.to_numeric(d.rating, errors='coerce')
    valid = d.rating.isin([1,2,3,4,5])
    audit['invalid_or_missing_ratings_removed'] = int((~valid).sum())
    d = d.loc[valid].copy()
    d['text'] = d.text.fillna('').astype(str).map(lambda s: re.sub(r'\s+', ' ', s).strip())
    valid = d.text.str.len().gt(0)
    audit['empty_text_rows_removed'] = int((~valid).sum())
    d = d.loc[valid].copy()
    d['date'] = pd.to_datetime(d.date, errors='coerce', utc=True, format='ISO8601')
    audit['invalid_dates_retained_as_missing'] = int(d.date.isna().sum())
    d['negative'] = d.rating.le(2).astype(int)
    d['word_count'] = d.text.str.split().str.len()
    # Identical normalized texts must remain in one partition.
    d['text_group'] = d.text.str.lower().map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    conflict = d.groupby('text_group').negative.transform('nunique').gt(1)
    audit['conflicting_text_rows_removed'] = int(conflict.sum())
    d = d.loc[~conflict].copy()
    audit['repeated_text_rows_removed'] = int(d.text_group.duplicated().sum())
    d = d.drop_duplicates('text_group').copy()
    audit['clean_rows'] = len(d)
    return d.reset_index(drop=True), audit

def split_data(d):
    # Union product groups and identical-text groups to avoid both types of overlap.
    parent = list(range(len(d)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a,b):
        parent[find(a)] = find(b)
    for col in ['product_id', 'text_group']:
        for ids in d.groupby(col).groups.values():
            ids = list(ids)
            for i in ids[1:]:
                union(ids[0], i)
    groups = np.array([find(i) for i in range(len(d))])
    if len(set(groups)) < 10:
        raise ValueError('Too few independent groups for a reliable split')
    first = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
    rest,test = next(first.split(d, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED+1)
    train_i,val_i = next(second.split(d.iloc[rest], groups=groups[rest]))
    parts = {'train':d.iloc[rest[train_i]].copy(), 'validation':d.iloc[rest[val_i]].copy(),
             'test':d.iloc[test].copy()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(parts[a].product_id) & set(parts[b].product_id)
        assert not set(parts[a].text_group) & set(parts[b].text_group)
    for name,p in parts.items():
        assert p.negative.nunique()==2, f'{name} needs both labels'
    return parts

def score(y,p,t):
    pred = p>=t
    return dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),
                recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),
                average_precision=average_precision_score(y,p),roc_auc=roc_auc_score(y,p))

def main():
    OUT.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
    raw = make_data(); raw.to_csv(DATA/'synthetic_reviews_raw.csv',index=False)
    d,audit = clean(raw); d.to_csv(DATA/'synthetic_reviews_clean.csv',index=False)
    parts = split_data(d)
    train,val,test = (parts[x] for x in ['train','validation','test'])
    candidates = {'Logistic regression':LogisticRegression(C=1,max_iter=1000,random_state=SEED),
                  'Multinomial naive Bayes':MultinomialNB(alpha=1)}
    fitted = {}; rows = []
    for name,clf in candidates.items():
        pipe = Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=10000)),('model',clf)])
        pipe.fit(train.text,train.negative)
        vp = pipe.predict_proba(val.text)[:,1]
        thresholds = np.arange(.10,.91,.05)
        threshold = float(max(thresholds,key=lambda t: f1_score(val.negative,vp>=t,zero_division=0)))
        fitted[name] = (pipe,threshold)
        rows.append({'model':name,'threshold':threshold,'validation_f1':f1_score(val.negative,vp>=threshold)})
    chosen = max(rows,key=lambda r:r['validation_f1'])['model']
    results = []
    for row in rows:
        pipe,t = fitted[row['model']]
        p = pipe.predict_proba(test.text)[:,1]
        results.append({**row,**score(test.negative,p,t)})
    dummy = DummyClassifier(strategy='prior').fit(np.zeros((len(train),1)),train.negative)
    dp = dummy.predict_proba(np.zeros((len(test),1)))[:,1]
    results.append({'model':'Prior baseline','threshold':.5,'validation_f1':None,**score(test.negative,dp,.5)})
    pd.DataFrame(results).to_csv(OUT/'model_metrics.csv',index=False)
    pipe,t = fitted[chosen]; p = pipe.predict_proba(test.text)[:,1]
    cm = confusion_matrix(test.negative,p>=t,labels=[0,1])
    # Conditional uncertainty on synthetic test examples only; not real-world confidence.
    rng = np.random.default_rng(SEED); boots=[]; y=test.negative.to_numpy()
    for _ in range(500):
        ix=rng.integers(0,len(y),len(y)); boots.append(f1_score(y[ix],p[ix]>=t,zero_division=0))
    ci = np.quantile(boots,[.025,.975]).tolist()
    summary = dict(provenance='SYNTHETIC DEMONSTRATION ONLY',seed=SEED,audit=audit,
                   splits={k:dict(rows=len(v),products=v.product_id.nunique(),negative_rate=float(v.negative.mean())) for k,v in parts.items()},
                   chosen_model=chosen,threshold=t,confusion_matrix=cm.tolist(),
                   conditional_f1_bootstrap_interval=ci,metrics=results,
                   versions={k:importlib.metadata.version(k) for k in ['pandas','numpy','scikit-learn','matplotlib']},python=platform.python_version())
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.size':10,'figure.dpi':140})
    fig,ax=plt.subplots(figsize=(6.4,3.3)); d.rating.value_counts().sort_index().plot.bar(ax=ax,color='#496a80')
    ax.set(title='Synthetic review ratings after cleaning',xlabel='Rating',ylabel='Reviews'); fig.tight_layout(); fig.savefig(OUT/'ratings.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,3.3)); d.groupby('category').negative.mean().plot.barh(ax=ax,color='#496a80')
    ax.set(title='Synthetic low rating share by category',xlabel='Share rated 1 or 2'); fig.tight_layout(); fig.savefig(OUT/'categories.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,3.6)); ConfusionMatrixDisplay(cm,display_labels=['Other','Low rating']).plot(ax=ax,colorbar=False,cmap='Blues')
    ax.set_title('Synthetic holdout confusion matrix'); fig.tight_layout(); fig.savefig(OUT/'confusion.png'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,3.3))
    for name,(model,_) in fitted.items():
        pr,rc,_=precision_recall_curve(test.negative,model.predict_proba(test.text)[:,1]); ax.plot(rc,pr,label=name)
    ax.axhline(test.negative.mean(),ls='--',color='gray',label='Positive prevalence')
    ax.set(title='Synthetic holdout precision and recall',xlabel='Recall',ylabel='Precision'); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT/'precision_recall.png'); plt.close(fig)
    # Save partition membership and prediction evidence for independent checking.
    pd.concat([v.assign(partition=k) for k,v in parts.items()]).to_csv(OUT/'partition_manifest.csv',index=False)
    test.assign(probability=p,predicted_negative=(p>=t).astype(int)).to_csv(OUT/'test_predictions.csv',index=False)
    (OUT/'cleaning_audit.json').write_text(json.dumps(audit,indent=2))
    (ROOT/'requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in summary['versions'].items())+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
