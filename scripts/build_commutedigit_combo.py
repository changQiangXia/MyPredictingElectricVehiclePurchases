"""Build candidates using only corrected commute digits plus quarter rule."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]; T='Will_Buy_EV'
base=pd.read_csv(ROOT/'artifacts/probe_publicbest_v19_high_wm010/submission.csv'); digit=pd.read_csv(ROOT/'artifacts/corrected_commute_digits_v1/submission.csv');tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
br=rankdata(base[T].to_numpy(float),method='average')/(len(base)+1.0);dr=rankdata(digit[T].to_numpy(float),method='average')/(len(digit)+1.0)
x=np.floor(tr.Daily_Commute_km.to_numpy(float)/.25)*.25
st=pd.DataFrame({'x':x,'y':tr.Will_Buy_EV.eq('Yes').to_numpy()}).groupby('x').y.agg(['size','sum'])
pure=st[(st['size']>=20)&(st['sum']==0)].index
xt=np.floor(te.Daily_Commute_km.to_numpy(float)/.25)*.25;cm=np.isin(xt,pure)
for dw in [-.10,-.15,-.20]:
 for mode in ['none','quarter_bottom']:
  p=(1-dw)*br+dw*dr
  if mode=='quarter_bottom':p[cm]=0.0
  name=f'nextday_commutedigit{int(abs(dw)*100):02d}_{mode}';out=ROOT/'artifacts'/name;out.mkdir(exist_ok=False);sub=base.copy();sub[T]=p;sub.to_csv(out/'submission.csv',index=False)
  meta={'run':name,'base':'probe_publicbest_v19_high_wm010','digit_member':'corrected_commute_digits_v1','digit_rank_weight':dw,'commute_rule': 'quarter-km pure-zero support>=20' if mode!='none' else None,'commute_rows':int(cm.sum()),'submitted':False,'valid_probability_range':bool(((p>=0)&(p<=1)).all()),'local_oof':'commute-only corrected digits negative residual +8.1e-6 at -10%; with quarter bottom +1.39e-5 at -10%, +1.61e-5 at -15%, +1.67e-5 at -20% on wave9','sha256':hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()}
  (out/'metrics.json').write_text(json.dumps(meta,indent=2)+'\n');print(name,meta['sha256'])
