import os, json, time, gc
os.environ['OMP_NUM_THREADS']='8'; os.environ['OPENBLAS_NUM_THREADS']='8'
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier
from train_encoded import ROOT,TARGET,features,digest
from train_xgb_formula_cv import margin_values
OUT=ROOT/'artifacts/experimental/full_te_teststyle_v1'; OUT.mkdir(parents=True,exist_ok=True)
def rk(x): return rankdata(x,method='average')/len(x)
tr,te=pd.read_csv(ROOT/'train.csv'),pd.read_csv(ROOT/'test.csv'); y=tr[TARGET].eq('Yes').to_numpy('int8'); base,keys,_=features(tr,te,False); X=base.to_numpy()[:len(tr)]; K=keys.to_numpy()[:len(tr)]; margin=margin_values(tr,'probit')
outer=np.load(ROOT/'artifacts/diagnosis_bagging_transfer_v1/outer_folds.npy'); results=[]
for f in [0,1,2]:
 dev=np.flatnonzero(outer!=f); hold=np.flatnonzero(outer==f)
 # full-dev TE (fit all dev labels), no holdout leakage
 partsA=[X[dev]]; partsB=[X[hold]]
 for smooth in ('auto',10.,100.):
  enc=TargetEncoder(target_type='binary',smooth=smooth,cv=5,shuffle=True,random_state=42)
  partsA.append(enc.fit_transform(K[dev],y[dev]).astype('float32')); partsB.append(enc.transform(K[hold]).astype('float32'))
 A=np.column_stack(partsA);B=np.column_stack(partsB)
 p=dict(device='cuda',tree_method='hist',max_bin=1024,n_estimators=6500,max_depth=5,learning_rate=.03,min_child_weight=10,subsample=.9,colsample_bytree=.85,reg_alpha=.071,reg_lambda=2.,objective='binary:logistic',eval_metric='auc',early_stopping_rounds=300,n_jobs=8,random_state=42)
 m=XGBClassifier(**p);m.fit(A,y[dev],base_margin=margin[dev],eval_set=[(B,y[hold])],base_margin_eval_set=[margin[hold]],verbose=False);m.set_params(device='cpu'); fp=m.predict_proba(B,base_margin=margin[hold])[:,1]
 bags=[]
 for j in range(10):
  q=np.load(ROOT/f'artifacts/diagnosis_bagging_transfer_v1/outer_{f}_seed_42_member_{j}.npz')['heldout_prediction']; bags.append(q)
 bag=np.mean(bags,axis=0)
 np.save(OUT/f'full_{f}.npy',fp)
 for w in [0,.1,.2,.3,.5,.7,1.0]:
  s=(1-w)*rk(bag)+w*rk(fp);results.append({'outer':f,'w_full':w,'bag_auc':roc_auc_score(y[hold],bag),'full_auc':roc_auc_score(y[hold],fp),'blend_auc':roc_auc_score(y[hold],s),'delta_vs_bag':roc_auc_score(y[hold],s)-roc_auc_score(y[hold],rk(bag))})
 print(f,results[-1])
 del A,B,m
pd.DataFrame(results).to_csv(OUT/'comparison.csv',index=False);rep=pd.DataFrame(results).groupby('w_full')[['bag_auc','full_auc','blend_auc','delta_vs_bag']].mean().reset_index();print(rep.to_string(index=False));json.dump({'run':'full_te_teststyle_v1','results':results,'summary':rep.to_dict('records'),'submitted':False},open(OUT/'report.json','w'),indent=2)
