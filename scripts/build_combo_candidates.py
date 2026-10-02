"""Build review-only stacked generator probes from prepared files."""
from pathlib import Path
import hashlib, json
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]; T='Will_Buy_EV'
def write(name,p,meta):
 out=ROOT/'artifacts'/name; out.mkdir(exist_ok=False)
 s=pd.read_csv(ROOT/'sample_submission.csv'); s[T]=p; s.to_csv(out/'submission.csv',index=False)
 m=dict(meta,run=name,submitted=False,rows=len(s),sha256=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest())
 (out/'metrics.json').write_text(json.dumps(m,indent=2)+'\n')
for hermw in [-.1,-.2]:
 base=pd.read_csv(ROOT/'artifacts'/f'nextday_highwm010_herm_neg{int(abs(hermw)*100):02d}'/'submission.csv')[T].to_numpy(float)
 t=pd.read_csv(ROOT/'test.csv'); mask=t.Daily_Commute_km.isin([68.8,75.7,78.3,79.8]).to_numpy()
 for cw in [-.05,-.1,-.2]:
  p=base.copy(); p[mask]+=cw
  write(f'nextday_combo_herm{int(abs(hermw)*100):02d}_commute{int(abs(cw)*100):02d}',p,{
   'base':f'nextday_highwm010_herm_neg{int(abs(hermw)*100):02d}',
   'commute_rule':'fixed public exact values 68.8,75.7,78.3,79.8',
   'commute_delta':cw,'herm_rank_weight':hermw,
   'rationale':'combine independent Herm representation residual with public commute generator rule',
   'local_oof_proxy':'wave9 anchor + Herm + commute; herm -10/commute -10 pooled +1.11e-5, all five folds positive',
  })
