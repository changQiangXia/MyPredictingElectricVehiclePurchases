"""Build rank-safe corrected-digit residual plus commute-rule candidates."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parent; T='Will_Buy_EV'
base=pd.read_csv(ROOT/'artifacts/probe_publicbest_v19_high_wm010/submission.csv'); digit=pd.read_csv(ROOT/'artifacts/corrected_digits_v1/submission.csv'); test=pd.read_csv(ROOT/'test.csv'); tr=pd.read_csv(ROOT/'train.csv')
br=rankdata(base[T].to_numpy(float),method='average')/(len(base)+1.0); dr=rankdata(digit[T].to_numpy(float),method='average')/(len(digit)+1.0)
x=np.floor(tr.Daily_Commute_km.to_numpy(float)/.25)*.25; st=pd.DataFrame({'x':x,'y':tr.Will_Buy_EV.eq('Yes').to_numpy()}).groupby('x').y.agg(['size','sum']); pure=st[(st['size']>=20)&(st['sum']==0)].index
xt=np.floor(test.Daily_Commute_km.to_numpy(float)/.25)*.25; cm=np.isin(xt,pure)
for dw in [-.05,-.10,-.15]:
 for mode,cw in [('none',0.0),('commute_bottom',None)]:
  pred=(1-dw)*br+dw*dr
  if mode=='commute_bottom': pred[cm]=0.0
  name=f'nextday_correcteddigit{int(abs(dw)*100):02d}_{mode}'
  out=ROOT/'artifacts'/name;out.mkdir(exist_ok=False);sub=base.copy();sub[T]=pred;sub.to_csv(out/'submission.csv',index=False)
  meta={'run':name,'base':'probe_publicbest_v19_high_wm010','digit_member':'corrected_digits_v1','digit_rank_weight':dw,'commute_rule':'quarter-km pure-zero support>=20' if mode!='none' else None,'commute_rows':int(cm.sum()),'submitted':False,'valid_probability_range':bool(((pred>=0)&(pred<=1)).all()),'local_oof':'corrected-digit negative residual robust +~5e-6 across 5 random 3-fold splits; with quarter commute bottom wave9 +1.26e-5 at dw=-0.10,cw=bottom','sha256':hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()};(out/'metrics.json').write_text(json.dumps(meta,indent=2)+'\n');print(name,meta['sha256'])
