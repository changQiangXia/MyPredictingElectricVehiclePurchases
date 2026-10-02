import os, json, time
os.environ['OMP_NUM_THREADS']='8'
import numpy as np,pandas as pd,xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
ROOT='.'; TARGET='Will_Buy_EV'
tr=pd.read_csv('train.csv'); te=pd.read_csv('test.csv'); y=tr[TARGET].eq('Yes').astype(float).to_numpy();
base_oof=np.load('artifacts/offline_wave9_seed_transfer15_v1/oof.npy'); base_test=np.load('artifacts/offline_wave9_seed_transfer15_v1/test.npy')
# compact feature view, plus anchor score
allx=pd.concat([tr.drop(columns=[TARGET]),te],ignore_index=True)
features=[]
for c in allx.columns:
 if c=='id': continue
 if allx[c].dtype=='object': features.append(pd.Categorical(allx[c].astype(str)).codes.astype('float32'))
 else: features.append(pd.to_numeric(allx[c],errors='coerce').fillna(-999).to_numpy(dtype='float32'))
X=np.column_stack(features); X=np.column_stack([X, np.r_[base_oof,base_test].astype('float32')]); n=len(tr)
# 3 folds, pseudo tails equal based on anchor test quantiles; total pseudo 10% train
sk=StratifiedKFold(3,shuffle=True,random_state=731); pred=np.zeros(n); rec=[]
for f,(ti,vi) in enumerate(sk.split(X[:n],y)):
 q=0.05; k=int(len(ti)*q); order=np.argsort(base_test); neg=order[:k]; pos=order[-k:]
 pi=np.r_[ti, n+neg, n+pos]; yy=np.r_[y[ti], np.full(k,0.02), np.full(k,0.98)]; ww=np.r_[np.ones(len(ti)), np.full(2*k,0.10)]
 dtr=xgb.DMatrix(X[pi],label=yy,weight=ww); dv=xgb.DMatrix(X[vi],label=y[vi]);
 params={'objective':'binary:logistic','eval_metric':'auc','tree_method':'hist','device':'cuda','max_depth':4,'eta':0.04,'subsample':0.85,'colsample_bytree':0.8,'min_child_weight':20,'reg_lambda':5,'seed':731+f}
 model=xgb.train(params,dtr,num_boost_round=800,evals=[(dv,'v')],early_stopping_rounds=60,verbose_eval=False)
 pred[vi]=model.predict(dv); rec.append({'fold':f,'auc':float(roc_auc_score(y[vi],pred[vi])),'best':model.best_iteration})
 print(rec[-1],flush=True)
auc=float(roc_auc_score(y,pred)); baseauc=float(roc_auc_score(y,base_oof)); print(json.dumps({'auc':auc,'base':baseauc,'delta':auc-baseauc,'folds':rec},indent=2))
json.dump({'auc':auc,'base':baseauc,'delta':auc-baseauc,'folds':rec},open('artifacts/experimental/softpl_pilot_v1.json','w'),indent=2)
