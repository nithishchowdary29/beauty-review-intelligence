"""Real public Amazon All Beauty analysis. No synthetic fallback or raw review redistribution.
Run: python src/run_project.py --source /path/to/All_Beauty.json.gz
Omit --source to download the documented archive into ignored data/raw/.
"""
from pathlib import Path
import argparse, gzip, json, hashlib, html, re, unicodedata, urllib.request
import importlib.metadata, platform, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
 average_precision_score, roc_auc_score, confusion_matrix, precision_recall_curve,
 ConfusionMatrixDisplay, brier_score_loss)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'; DATA=ROOT/'data'
URL='https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/All_Beauty.json.gz'
META_URL='https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_All_Beauty.json.gz'
META_SHA='f2da6049903fc132cbcf9a45c03128ab65ba7140e29691469ecd8ce839612ea3'
EXPECTED_SHA='a430509b38ff244fb75a84213e82ae64dc735158587687c80e86465d286bf22d'
SEED=42; SAMPLE_SIZE=30000
NUMERIC=['log_words','exclam_per_100','question_per_100','uppercase_share']

def write_json(path,obj):
 path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def normalize(value):
 if not isinstance(value,str):return ''
 s=unicodedata.normalize('NFKC',html.unescape(value))
 s=re.sub(r'<[^>]+>',' ',s)
 return re.sub(r'\s+',' ',s).strip()

def acquire(path,metadata_path=None):
 if path is None:
  path=DATA/'raw/All_Beauty.json.gz';path.parent.mkdir(parents=True,exist_ok=True)
  if not path.exists():
   temp=path.with_suffix('.part')
   with urllib.request.urlopen(URL,timeout=90) as r,temp.open('wb') as f:
    while True:
     block=r.read(1024*1024)
     if not block:break
     f.write(block)
   temp.replace(path)
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 if digest!=EXPECTED_SHA:
  raise ValueError('Source checksum changed. Review dataset version before proceeding.')
 if metadata_path is None:
  metadata_path=DATA/'raw/meta_All_Beauty.json.gz';metadata_path.parent.mkdir(parents=True,exist_ok=True)
  if not metadata_path.exists():
   with urllib.request.urlopen(META_URL,timeout=90) as r:metadata_path.write_bytes(r.read())
 meta_digest=hashlib.sha256(metadata_path.read_bytes()).hexdigest()
 if meta_digest!=META_SHA:raise ValueError('Metadata checksum changed')
 metadata={};meta_rows=0
 for line in gzip.open(metadata_path,'rt',encoding='utf-8'):
  item=json.loads(line);metadata[item['asin']]=item;meta_rows+=1
 # Uniform reservoir sample of source rows; no label based selection.
 rng=np.random.default_rng(SEED);sample=[];ratings={};items=set();dates=[]
 for i,line in enumerate(gzip.open(path,'rt',encoding='utf-8')):
  x=json.loads(line);rating=x.get('overall');ratings[str(rating)]=ratings.get(str(rating),0)+1
  items.add(x.get('asin'))
  if x.get('unixReviewTime') is not None:dates.append(x['unixReviewTime'])
  # Local private identity used only to quantify author overlap, never exported.
  row={'source_row':i+1,'product_id':x.get('asin'),'rating':rating,
       'text':x.get('reviewText'),'timestamp':x.get('unixReviewTime'),
       'verified':x.get('verified'),'author':x.get('reviewerID'),
       'metadata_match':bool(normalize(metadata.get(x.get('asin'),{}).get('title','')))}
  if i<SAMPLE_SIZE:sample.append(row)
  else:
   j=int(rng.integers(0,i+1))
   if j<SAMPLE_SIZE:sample[j]=row
 sample=sorted(sample,key=lambda x:x['source_row'])
 manifest=dict(dataset='Amazon Review Data 2018 All Beauty full category',source_url=URL,
  publisher_documentation='https://nijianmo.github.io/amazon/index.html',
  acquired_utc=datetime.now(timezone.utc).isoformat(),sha256=digest,compressed_bytes=path.stat().st_size,
  source_rows=i+1,source_products=len(items),source_ratings=ratings,
  source_date_min=pd.to_datetime(min(dates),unit='s',utc=True).isoformat(),
  source_date_max=pd.to_datetime(max(dates),unit='s',utc=True).isoformat(),
  sampled_rows=len(sample),seed=SEED,sampling='Uniform row reservoir over complete archive in source order',
  metadata_url=META_URL,metadata_sha256=meta_digest,metadata_rows=meta_rows,metadata_unique_asins=len(metadata),
  metadata_duplicate_asins=meta_rows-len(metadata),
  raw_data_redistributed=False,notes='Historical public research data. No live scraping. No broad redistribution license asserted. Metadata matching is a coverage check, not complete semantic domain validation.')
 return pd.DataFrame(sample),manifest

def clean(raw):
 d=raw.copy();audit={'sampled_rows':len(d),'sample_missing_text':int(d.text.isna().sum()),
 'sample_missing_rating':int(d.rating.isna().sum()),'sample_missing_product':int(d.product_id.isna().sum())}
 d['rating']=pd.to_numeric(d.rating,errors='coerce')
 valid=d.rating.isin([1,2,3,4,5]);audit['invalid_labels_removed']=int((~valid).sum());d=d.loc[valid].copy()
 valid=d.product_id.fillna('').astype(str).str.len().gt(0);audit['missing_product_removed']=int((~valid).sum());d=d.loc[valid].copy()
 audit['unmatched_metadata_removed']=int((~d.metadata_match).sum());d=d.loc[d.metadata_match].copy()
 d['text']=d.text.map(normalize);valid=d.text.str.len().gt(0);audit['empty_text_removed']=int((~valid).sum());d=d.loc[valid].copy()
 d['date']=pd.to_datetime(d.timestamp,unit='s',errors='coerce',utc=True)
 audit['invalid_dates_retained']=int(d.date.isna().sum())
 d['negative']=d.rating.le(2).astype(int)
 d['text_hash']=d.text.str.casefold().map(lambda s:hashlib.sha256(s.encode()).hexdigest())
 # Deduplication is label independent: retain earliest source row, even if labels disagree.
 audit['duplicate_text_rows']=int(d.text_hash.duplicated().sum())
 audit['text_groups_with_conflicting_binary_labels']=int(d.groupby('text_hash').negative.nunique().gt(1).sum())
 audit['negative_before_dedup']=int(d.negative.sum());d=d.drop_duplicates('text_hash',keep='first').copy()
 d['word_count']=d.text.str.split().str.len();d['log_words']=np.log1p(d.word_count)
 length=d.text.str.len().clip(lower=1)
 d['exclam_per_100']=100*d.text.str.count('!')/length
 d['question_per_100']=100*d.text.str.count(r'\?')/length
 d['uppercase_share']=d.text.map(lambda s:sum(c.isupper() for c in s)/max(1,sum(c.isalpha() for c in s)))
 audit['clean_rows']=len(d);audit['negative_after_dedup']=int(d.negative.sum())
 audit['clean_products']=d.product_id.nunique()
 audit['word_count_p99']=float(d.word_count.quantile(.99));audit['word_count_max']=int(d.word_count.max())
 assert len(d)==len(raw)-audit['invalid_labels_removed']-audit['missing_product_removed']-audit['unmatched_metadata_removed']-audit['empty_text_removed']-audit['duplicate_text_rows']
 return d.reset_index(drop=True),audit

def partitions(d):
 rest,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED).split(d,groups=d.product_id))
 tri,vai=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=SEED+1).split(d.iloc[rest],groups=d.iloc[rest].product_id))
 parts={'train':d.iloc[rest[tri]].copy(),'validation':d.iloc[rest[vai]].copy(),'test':d.iloc[test].copy()}
 for a,b in [('train','validation'),('train','test'),('validation','test')]:
  assert not set(parts[a].product_id)&set(parts[b].product_id)
  assert not set(parts[a].text_hash)&set(parts[b].text_hash)
 for v in parts.values():assert v.negative.nunique()==2
 return parts

def pipeline(kind,C=1,alpha=1):
 vector=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_df=.98,max_features=30000,sublinear_tf=True)
 if kind=='nb':return Pipeline([('features',ColumnTransformer([('text',vector,'text')],sparse_threshold=1)),('model',MultinomialNB(alpha=alpha))])
 fields=[('text',vector,'text')]
 if kind=='hybrid':fields.append(('numeric',StandardScaler(),NUMERIC))
 return Pipeline([('features',ColumnTransformer(fields,sparse_threshold=1)),('model',LogisticRegression(C=C,class_weight='balanced',solver='liblinear',max_iter=1000,random_state=SEED))])

def metrics(y,prob,t):
 pred=prob>=t
 return dict(accuracy=float(accuracy_score(y,pred)),precision=float(precision_score(y,pred,zero_division=0)),
  recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),
  average_precision=float(average_precision_score(y,prob)),roc_auc=float(roc_auc_score(y,prob)),
  brier=float(brier_score_loss(y,prob)))

def export_safe(d,path):
 # Only numeric/structural evidence; no review text, titles, author identity or scores derived from author.
 cols=[c for c in d if c not in ['text','author','timestamp']]
 d[cols].to_csv(path,index=False,compression={'method':'gzip','mtime':0})

def figsave(name):
 plt.tight_layout();plt.savefig(OUT/name,dpi=150,bbox_inches='tight');plt.close()

def eda(train):
 rating=train.rating.value_counts().sort_index().rename_axis('rating').reset_index(name='count');rating.to_csv(OUT/'training_rating_counts.csv',index=False)
 plt.figure(figsize=(6.8,3.5));plt.bar(rating.rating.astype(int).astype(str),rating['count'],color='#42677f');plt.title('Real reviews in training set by rating');plt.xlabel('Star rating');plt.ylabel('Reviews');figsave('ratings.png')
 plt.figure(figsize=(6.8,3.6));plt.boxplot([np.log1p(train.loc[train.negative==v,'word_count']) for v in [0,1]],tick_labels=['Other ratings 3 to 5','Low ratings 1 to 2'],showfliers=False);plt.ylabel('log1p word count');plt.title('Review length by rating label in training data');figsave('length.png')
 counts=train.product_id.value_counts().head(10);counts.to_csv(OUT/'top_training_products.csv',header=['reviews'])
 plt.figure(figsize=(6.8,3.7));counts.sort_values().plot.barh(color='#42677f');plt.xlabel('Training reviews');plt.title('Ten most represented training products');figsave('products.png')
 terms={'packaging':r'\b(?:packag\w*|bottle|pump|leak\w*)\b','fragrance':r'\b(?:scent\w*|smell\w*|fragrance|perfume)\b','value':r'\b(?:price|expensive|cheap|money|worth)\b','texture':r'\b(?:greasy|sticky|oily|texture|absor\w*)\b'}
 theme=[]
 for name,pattern in terms.items():
  flag=train.text.str.contains(pattern,case=False,regex=True);n=int(flag.sum());neg=int(train.loc[flag,'negative'].sum())
  theme.append(dict(theme=name,mentions=n,negative_mentions=neg,low_rating_share=neg/n if n else 0))
 td=pd.DataFrame(theme);td.to_csv(OUT/'theme_mentions.csv',index=False)
 plt.figure(figsize=(6.8,3.5));plt.bar(td.theme,td.mentions,color='#42677f');plt.ylabel('Reviews mentioning terms');plt.title('Exploratory term mentions in training text');figsave('themes.png')
 low=train.loc[train.negative==1,'word_count'];other=train.loc[train.negative==0,'word_count']
 u,p=stats.mannwhitneyu(low,other,alternative='two-sided');effect=2*u/(len(low)*len(other))-1
 info=dict(training_rows=len(train),training_products=train.product_id.nunique(),negative_count=int(train.negative.sum()),
  negative_rate=float(train.negative.mean()),length_by_label={str(k):{'n':len(g),'median':float(g.word_count.median()),'q25':float(g.word_count.quantile(.25)),'q75':float(g.word_count.quantile(.75))} for k,g in train.groupby('negative')},
  length_test={'method':'Exploratory row-level Mann Whitney U','U':float(u),'pvalue':float(p),'rank_biserial_low_vs_other':float(effect),'caveat':'Reviews share products and authors; iid p value is descriptive and not confirmatory.'},
  top10_product_share=float(counts.sum()/len(train)),themes=theme)
 return info

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);parser.add_argument('--metadata',type=Path);args=parser.parse_args()
 OUT.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
 t0=time.time();raw,manifest=acquire(args.source,args.metadata);d,audit=clean(raw);parts=partitions(d)
 train,val,test=(parts[k] for k in ['train','validation','test'])
 # Save only non-text evidence; users can regenerate text locally from source.
 export_safe(d,DATA/'clean_features.csv.gz');export_safe(pd.concat([v.assign(partition=k) for k,v in parts.items()]),OUT/'partition_manifest.csv.gz')
 write_json(DATA/'source_manifest.json',manifest);write_json(OUT/'cleaning_audit.json',audit)
 info=eda(train);write_json(OUT/'eda_summary.json',info)
 pd.DataFrame([(k,raw[k].isna().sum(),str(raw[k].dtype)) for k in raw],columns=['field','missing','dtype']).to_csv(OUT/'source_sample_profile.csv',index=False)
 folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=44).split(train,train.negative,train.product_id))
 for tr,va in folds:assert not set(train.iloc[tr].product_id)&set(train.iloc[va].product_id)
 cv=[];choices={}
 for kind,label,params in [('text','Logistic text',[.5,1,2]),('nb','Naive Bayes',[.5,1]),('hybrid','Logistic text plus style',[.5,1,2])]:
  for param in params:
   model=pipeline(kind,alpha=param) if kind=='nb' else pipeline(kind,C=param)
   scores=cross_val_score(model,train,train.negative,cv=folds,scoring='average_precision',n_jobs=1)
   cv.append(dict(model=label,parameter=param,mean_ap=float(scores.mean()),std_ap=float(scores.std(ddof=1)),fold_1=float(scores[0]),fold_2=float(scores[1]),fold_3=float(scores[2])))
  chosen=max([r for r in cv if r['model']==label],key=lambda x:x['mean_ap']);choices[label]=(kind,chosen['parameter'])
 pd.DataFrame(cv).to_csv(OUT/'cross_validation.csv',index=False)
 fitted={};validation=[];threshold_rows=[]
 for label,(kind,param) in choices.items():
  model=pipeline(kind,alpha=param) if kind=='nb' else pipeline(kind,C=param)
  model.fit(train,train.negative);vp=model.predict_proba(val)[:,1]
  for threshold in np.arange(.05,.951,.025):
   threshold_rows.append(dict(model=label,threshold=float(threshold),**metrics(val.negative,vp,float(threshold))))
  best=max([r for r in threshold_rows if r['model']==label],key=lambda r:r['f1'])
  threshold=best['threshold'];fitted[label]=(model,threshold)
  validation.append(dict(model=label,parameter=param,threshold=threshold,**metrics(val.negative,vp,threshold)))
 pd.DataFrame(validation).to_csv(OUT/'validation_metrics.csv',index=False);pd.DataFrame(threshold_rows).to_csv(OUT/'validation_thresholds.csv',index=False)
 # Select by validation average precision; F1 threshold is set separately on validation only.
 chosen=max(validation,key=lambda r:r['average_precision'])['model']
 results=[];probabilities={};y=test.negative.to_numpy()
 for label,(model,threshold) in fitted.items():
  prob=model.predict_proba(test)[:,1];probabilities[label]=prob
  results.append(dict(model=label,threshold=threshold,**metrics(y,prob,threshold)))
 for label,prob,threshold in [('Majority baseline',np.full(len(test),train.negative.mean()),.5),('All flagged baseline',np.ones(len(test)),.5)]:
  results.append(dict(model=label,threshold=threshold,**metrics(y,prob,threshold)))
 pd.DataFrame(results).to_csv(OUT/'model_metrics.csv',index=False)
 model,threshold=fitted[chosen];prob=probabilities[chosen];cm=confusion_matrix(y,prob>=threshold,labels=[0,1])
 predictions=test[['source_row','product_id','rating','negative','word_count','text_hash']].copy();predictions['score']=prob;predictions['predicted_negative']=(prob>=threshold).astype(int)
 predictions.to_csv(OUT/'test_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 # Conditional product-cluster bootstrap. Fits and threshold are frozen.
 rng=np.random.default_rng(45);groups=list(test.groupby('product_id').indices.values());boots=[]
 for _ in range(500):
  ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
  if len(np.unique(y[ix]))<2:continue
  boots.append(metrics(y[ix],prob[ix],threshold))
 intervals={k:[float(x) for x in np.quantile([b[k] for b in boots],[.025,.975])] for k in ['precision','recall','f1','average_precision']}
 plt.figure(figsize=(6.8,3.8))
 for label,pr in probabilities.items():
  precision,recall,_=precision_recall_curve(y,pr);plt.plot(recall,precision,label=label)
 plt.axhline(y.mean(),ls='--',color='gray',label='Test low rating prevalence');plt.xlabel('Recall');plt.ylabel('Precision');plt.title('Real review holdout precision and recall');plt.legend(fontsize=8);figsave('precision_recall.png')
 fig,ax=plt.subplots(figsize=(5.5,3.8));ConfusionMatrixDisplay(cm,display_labels=['Other','Low rating']).plot(ax=ax,cmap='Blues',colorbar=False);ax.set_title('Real review holdout confusion matrix');figsave('confusion.png')
 mt=pd.DataFrame(results).set_index('model')
 mt[['precision','recall','f1']].plot.bar(figsize=(7,4),color=['#42677f','#92adb8','#b98059']);plt.ylim(0,1);plt.xticks(rotation=15,ha='right');plt.title('Measured holdout model comparison');figsave('model_comparison.png')
 tv=pd.DataFrame(threshold_rows);tv=tv[tv.model==chosen]
 plt.figure(figsize=(6.8,3.5))
 for key in ['precision','recall','f1']:plt.plot(tv.threshold,tv[key],label=key)
 plt.axvline(threshold,color='gray',ls='--');plt.xlabel('Threshold');plt.ylabel('Validation metric');plt.title('Threshold chosen on validation data');plt.legend();figsave('thresholds.png')
 # Readable coefficient associations on the fitted text-only model.
 lr=fitted['Logistic text'][0];vector=lr.named_steps['features'].named_transformers_['text'];terms=vector.get_feature_names_out();weights=lr.named_steps['model'].coef_[0]
 ix=np.r_[np.argsort(weights)[:12],np.argsort(weights)[-12:]]
 pd.DataFrame({'term':terms[ix],'coefficient':weights[ix]}).to_csv(OUT/'coefficient_terms.csv',index=False)
 # Export error signatures, not copyrighted review excerpts or reviewer identities.
 errors=predictions[predictions.negative!=predictions.predicted_negative].copy();errors['error_type']=np.where(errors.negative==1,'false negative','false positive')
 errors.head(30).to_csv(OUT/'error_cases.csv',index=False)
 author_overlap=len(set(train.author.dropna())&set(test.author.dropna()))
 summary=dict(status='Executed on real public reviews',seed=SEED,sample_size=SAMPLE_SIZE,
  dataset=manifest,audit=audit,eda=info,
  splits={k:dict(rows=len(v),products=v.product_id.nunique(),low_rating_count=int(v.negative.sum()),low_rating_rate=float(v.negative.mean())) for k,v in parts.items()},
  train_test_shared_authors=author_overlap,selected_model=chosen,selected_threshold=threshold,
  metrics=results,validation=validation,confusion_matrix=cm.tolist(),bootstrap_intervals=intervals,
  bootstrap_replicates=len(boots),selection_rule='Training group CV AP for hyperparameters then validation AP for model and validation F1 for threshold',
  no_product_overlap=True,no_exact_text_overlap=True,
  runtime_seconds=round(time.time()-t0,2),python=platform.python_version(),
  versions={m:importlib.metadata.version(m) for m in ['numpy','pandas','scipy','scikit-learn','matplotlib']})
 write_json(OUT/'summary.json',summary)
 (ROOT/'requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in summary['versions'].items())+'\n')
 print(json.dumps({'source':manifest,'audit':audit,'splits':summary['splits'],'chosen':chosen,'metrics':results,'intervals':intervals,'runtime_seconds':summary['runtime_seconds']},indent=2))

if __name__=='__main__':main()
