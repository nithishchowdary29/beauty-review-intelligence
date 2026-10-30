"""Independent checks of published evidence. Run after run_project.py."""
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import precision_score,recall_score,f1_score,average_precision_score,confusion_matrix
R=Path(__file__).resolve().parents[1];O=R/'outputs'
s=json.loads((O/'summary.json').read_text());a=s['audit']
assert a['sampled_rows']-a['invalid_labels_removed']-a['missing_product_removed']-a['unmatched_metadata_removed']-a['empty_text_removed']-a['duplicate_text_rows']==a['clean_rows']
p=pd.read_csv(O/'partition_manifest.csv.gz');t=pd.read_csv(O/'test_predictions.csv.gz')
assert len(p)==a['clean_rows']
assert p.groupby('product_id').partition.nunique().max()==1
assert p.groupby('text_hash').partition.nunique().max()==1
assert p.text_hash.is_unique
assert p.rating.isin([1,2,3,4,5]).all()
assert (p.negative==p.rating.le(2)).all()
assert set(t.source_row)==set(p.loc[p.partition=='test','source_row'])
assert confusion_matrix(t.negative,t.predicted_negative,labels=[0,1]).tolist()==s['confusion_matrix']
selected=next(x for x in s['metrics'] if x['model']==s['selected_model'])
for key,actual in [('precision',precision_score(t.negative,t.predicted_negative)),('recall',recall_score(t.negative,t.predicted_negative)),('f1',f1_score(t.negative,t.predicted_negative)),('average_precision',average_precision_score(t.negative,t.score))]:
 assert abs(actual-selected[key])<1e-12,key
v=pd.read_csv(O/'validation_metrics.csv')
assert v.loc[v.average_precision.idxmax(),'model']==s['selected_model']
assert (t.predicted_negative==(t.score>=s['selected_threshold']).astype(int)).all()
assert not any(c in p for c in ['text','reviewerName','reviewerID','author'])
result={'status':'PASS','checks':['cleaning reconciliation','valid label mapping','product separation','exact text separation','holdout membership','confusion matrix','recomputed metrics','validation selection','threshold consistency','no reviewer identities or raw text in exported data'],'clean_rows':len(p),'test_rows':len(t)}
(O/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
