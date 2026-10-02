"""Independent low-variance feature view: rank/transform + one-hot logistic."""
import os
os.environ['OMP_NUM_THREADS']='8'; os.environ['OPENBLAS_NUM_THREADS']='8'
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler,QuantileTransformer,PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
def make(df):
 d=df.drop(columns=['id','Will_Buy_EV'],errors='ignore').copy()
 d['income_log']=np.log1p(d.Annual_Income_USD)
 d['income_sqrt']=np.sqrt(d.Annual_Income_USD)
 d['commute_log']=np.log1p(d.Daily_Commute_km)
 d['stations_total']=d.Charging_Stations_Near_Home+d.Charging_Stations_Near_Work
 d['subsidy']=(d.Subsidy_Available=='Yes').astype(int)
 d['home_charge']=(d.Home_Charging_Possible=='Yes').astype(int)
 d['high_anxiety']=(d.Range_Anxiety_Level=='High').astype(int)
 d['formula']=1.2*d.Annual_Income_USD/1e5+.6*d.Environmental_Concern_Level+2*d.subsidy-d.Range_Anxiety_Level.map({'Low':0,'Medium':1,'High':3}).fillna(1)
 return d
def main():
 tr=pd.read_csv(ROOT/'train.csv'); te=pd.read_csv(ROOT/'test.csv'); y=tr.Will_Buy_EV.eq('Yes').to_numpy(); X=make(tr); Xt=make(te)
 cats=X.select_dtypes('object').columns.tolist(); nums=[c for c in X if c not in cats]
 pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median'),QuantileTransformer(n_quantiles=256,output_distribution='normal',subsample=100000,random_state=42),StandardScaler()),nums),('cat',OneHotEncoder(handle_unknown='ignore'),cats)])
 model=make_pipeline(pre,LogisticRegression(C=.5,max_iter=100,solver='lbfgs'))
 o=np.zeros(len(tr)); p=np.zeros(len(te)); scores=[]; start=time.time()
 for f,(ti,vi) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
  model.fit(X.iloc[ti],y[ti]); o[vi]=model.predict_proba(X.iloc[vi])[:,1]; p+=model.predict_proba(Xt)[:,1]/5; scores.append(float(roc_auc_score(y[vi],o[vi]))); print(json.dumps({'fold':f,'auc':scores[-1]}),flush=True)
 out=ROOT/'artifacts'/'rank_logistic_v1'; out.mkdir(exist_ok=True); np.save(out/'oof.npy',o); np.save(out/'test.npy',p); pd.DataFrame({'id':te.id,'Will_Buy_EV':p}).to_csv(out/'submission.csv',index=False)
 m={'run':'rank_logistic_v1','oof_auc':float(roc_auc_score(y,o)),'fold_auc':scores,'fold_std':float(np.std(scores)),'status':'complete','elapsed':time.time()-start}; (out/'metrics.json').write_text(json.dumps(m,indent=2)+'\n'); print(json.dumps(m,indent=2))
if __name__=='__main__': main()
