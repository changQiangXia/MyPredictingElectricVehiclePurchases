import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
ROOT='artifacts'; y=pd.read_csv('train.csv').Will_Buy_EV.eq('Yes').to_numpy(); rows=[]
def rk(x):return rankdata(x,method='average')/len(x)
for f in [0,1,2]:
 z=np.load(f'{ROOT}/experimental/modulo_transfer_audit_v1/outer_{f}.npz'); hold=z['heldout_indices']; lab=y[hold]; bag=z['xgb_bag10']; full=np.load(f'{ROOT}/experimental/full_te_teststyle_v1/full_{f}.npy'); mod=z['mod1000_s500_full_outer80'] if 'mod1000_s500_full_outer80' in z.files else z['mod1000_s100_full_outer80']
 # saved audit has s2/s100 only, use s100 clean-ish
 for wf in [.1,.2,.3]:
  for wm in [.0,.002,.004,.006,.008]:
   if wf+wm>=1:continue
   s=(1-wf-wm)*rk(bag)+wf*rk(full)+wm*rk(mod)
   rows.append({'f':f,'wf':wf,'wm':wm,'auc':roc_auc_score(lab,s),'delta':roc_auc_score(lab,s)-roc_auc_score(lab,rk(bag))})
rep=pd.DataFrame(rows); print(rep.groupby(['wf','wm']).delta.mean().sort_values(ascending=False).head(20)); rep.to_csv(ROOT+'/experimental/full_te_mod_combo.csv',index=False)
