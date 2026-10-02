"""Independent fold-seed audit of the fixed quarter-km pure-zero family."""
from pathlib import Path
import json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
train=pd.read_csv(ROOT/'train.csv'); y=train.Will_Buy_EV.eq('Yes').to_numpy(np.int8); x=np.floor(train.Daily_Commute_km.to_numpy(float)/.25)*.25
folds=StratifiedKFold(5,shuffle=True,random_state=2026); anchors={n:np.load(ROOT/'artifacts'/n/'oof.npy') for n in ['offline_wave5_conservative','offline_wave9_seed_transfer15_v1']}; out=[]
for minimum in [10, 20]:
 for name,a in anchors.items():
  r=rankdata(a,method='average')/(len(a)+1.0); p=np.empty_like(r); ds=[]; hits=[]; pos=[]
  for fold,(ti,vi) in enumerate(folds.split(np.zeros(len(y)),y)):
   stats=pd.DataFrame({'k':x[ti],'y':y[ti]}).groupby('k').y.agg(['size','sum']); pure=stats[(stats['size']>=minimum)&(stats['sum']==0)].index
   m=np.isin(x[vi],pure);p[vi]=r[vi];p[vi[m]]=0.; ds.append(float(roc_auc_score(y[vi],p[vi])-roc_auc_score(y[vi],r[vi])));hits.append(int(m.sum()));pos.append(int(y[vi][m].sum()))
  out.append({'minimum':minimum,'anchor':name,'pooled_gain':float(roc_auc_score(y,p)-roc_auc_score(y,r)),'fold_gains':ds,'positive_folds':int(sum(d>0 for d in ds)),'rows':hits,'positives':pos})
path=ROOT/'artifacts/quarter_rule_seed2026_audit.json';path.write_text(json.dumps({'family':'floor commute .25km, min support20, pure zero','results':out,'interpretation':'second fold seed audit; labels reused from competition, not independent test evidence'},indent=2)+'\n');print(json.dumps(out,indent=2))
