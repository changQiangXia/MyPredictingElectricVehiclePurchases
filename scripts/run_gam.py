"""Spline + one-hot logistic model: a deliberately different, interpretable member."""
import os
os.environ['OMP_NUM_THREADS']='8'
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
def main():
 tr=pd.read_csv(ROOT/'train.csv'); te=pd.read_csv(ROOT/'test.csv'); y=tr.Will_Buy_EV.eq('Yes').to_numpy()
 cols=[c for c in te if c!='id']; cats=[c for c in cols if tr[c].dtype=='object']; nums=[c for c in cols if c not in cats]
 # Nonlinear basis on continuous drivers, one-hot for categorical variables.
 pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),SplineTransformer(n_knots=12,degree=3),StandardScaler(with_mean=False)),nums),('cat',OneHotEncoder(handle_unknown='ignore'),cats)])
 model=make_pipeline(pre,LogisticRegression(C=0.8,max_iter=300,solver='saga',n_jobs=8))
 X=tr[cols]; Xt=te[cols]; oof=np.zeros(len(tr)); pred=np.zeros(len(te)); scores=[]; start=time.time()
 folds=StratifiedKFold(5,shuffle=True,random_state=42)
 for f,(ti,vi) in enumerate(folds.split(X,y)):
  model.fit(X.iloc[ti],y[ti]); vp=model.predict_proba(X.iloc[vi])[:,1]; tp=model.predict_proba(Xt)[:,1]
  oof[vi]=vp; pred+=tp/5; scores.append(float(roc_auc_score(y[vi],vp))); print(json.dumps({'fold':f,'auc':scores[-1]}),flush=True)
 out=ROOT/'artifacts'/'gam_logistic_v1'; out.mkdir(exist_ok=True); np.save(out/'oof.npy',oof); np.save(out/'test.npy',pred)
 pd.DataFrame({'id':te.id,'Will_Buy_EV':pred}).to_csv(out/'submission.csv',index=False)
 m={'run':'gam_logistic_v1','oof_auc':float(roc_auc_score(y,oof)),'fold_auc':scores,'fold_std':float(np.std(scores)),'status':'complete','elapsed':time.time()-start}
 (out/'metrics.json').write_text(json.dumps(m,indent=2)+'\n'); print(json.dumps(m,indent=2))
if __name__=='__main__':main()
