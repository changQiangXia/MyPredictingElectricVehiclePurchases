"""Ten-fold nested target-encoding XGBoost; predictions are not mixed with 5-fold runs."""
import os
os.environ['OMP_NUM_THREADS']='8'; os.environ['OPENBLAS_NUM_THREADS']='8'
import hashlib,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
import train_encoded as base

ROOT=Path(__file__).resolve().parent
def main():
 import argparse
 ap=argparse.ArgumentParser(); ap.add_argument('--run',required=True); ap.add_argument('--depth',type=int,default=5); ap.add_argument('--learning-rate',type=float,default=.03); ap.add_argument('--iterations',type=int,default=6500); ap.add_argument('--model-seed',type=int,default=42); ap.add_argument('--original',action='store_true'); args=ap.parse_args()
 out=ROOT/'artifacts'/args.run; out.mkdir(parents=True,exist_ok=True); start=time.time()
 tr,te=pd.read_csv(ROOT/'train.csv'),pd.read_csv(ROOT/'test.csv'); y=tr.Will_Buy_EV.eq('Yes').to_numpy(dtype='int8'); n=len(tr); kfold=10
 fold_path=ROOT/'artifacts'/'folds_seed42_10.npy'
 if fold_path.exists(): folds=np.load(fold_path)
 else:
  folds=np.full(n,-1,dtype='int8')
  for f,(_,vi) in enumerate(StratifiedKFold(kfold,shuffle=True,random_state=42).split(tr,y)): folds[vi]=f
  np.save(fold_path,folds)
 assert set(folds)==set(range(kfold))
 X,key,_=base.features(tr,te,args.original); Xt=X.iloc[n:].to_numpy(); X=X.iloc[:n].to_numpy(); K=key.iloc[:n].to_numpy(); Kt=key.iloc[n:].to_numpy(); names=list(base.features(tr,te,args.original)[0].columns)
 for s in ('auto',10.,100.): names += [f'{c}_TE_{s}' for c in key]
 oof=np.full(n,np.nan); pred=np.zeros(len(te)); records=[]
 for fold in range(kfold):
  fs=time.time(); ti=np.flatnonzero(folds!=fold); vi=np.flatnonzero(folds==fold); ck=out/f'fold_{fold}.npz'
  if ck.exists():
   z=np.load(ck); oof[vi]=z['valid_prediction']; pred+=z['test_prediction']/kfold; records.append(json.loads((out/f'fold_{fold}.json').read_text())); print(json.dumps({'event':'resumed','fold':fold}),flush=True); continue
  A=[X[ti]]; B=[X[vi]]; C=[Xt]; encoders=[]
  for s in ('auto',10.,100.):
   enc=TargetEncoder(target_type='binary',smooth=s,cv=5,shuffle=True,random_state=42)
   A.append(enc.fit_transform(K[ti],y[ti]).astype('float32')); B.append(enc.transform(K[vi]).astype('float32')); C.append(enc.transform(Kt).astype('float32')); encoders.append(enc)
  A=np.concatenate(A,1); B=np.concatenate(B,1); C=np.concatenate(C,1)
  from xgboost import XGBClassifier
  model=XGBClassifier(device='cuda',tree_method='hist',max_bin=1024,n_estimators=args.iterations,max_depth=args.depth,learning_rate=args.learning_rate,min_child_weight=10,subsample=.9,colsample_bytree=.85,reg_alpha=.071,reg_lambda=2.,objective='binary:logistic',eval_metric='auc',early_stopping_rounds=300,n_jobs=8,random_state=args.model_seed)
  model.fit(A,y[ti],eval_set=[(B,y[vi])],verbose=1000); vp=model.predict_proba(B)[:,1]; tp=model.predict_proba(C)[:,1]; oof[vi]=vp; pred+=tp/kfold
  rec={'fold':fold,'auc':float(roc_auc_score(y[vi],vp)),'best_iteration':int(model.best_iteration)+1,'seconds':time.time()-fs}; records.append(rec); np.savez_compressed(ck,valid_prediction=vp,test_prediction=tp); (out/f'fold_{fold}.json').write_text(json.dumps(rec,indent=2)+'\n'); joblib.dump(encoders,out/f'encoders_{fold}.joblib'); print(json.dumps(rec),flush=True)
  del A,B,C,model,encoders
 assert np.isfinite(oof).all(); np.save(out/'oof.npy',oof); np.save(out/'test.npy',pred); pd.DataFrame({'id':te.id,'Will_Buy_EV':pred}).to_csv(out/'submission.csv',index=False)
 m={'run':args.run,'n_splits':kfold,'oof_auc':float(roc_auc_score(y,oof)),'folds':records,'fold_mean':float(np.mean([r['auc'] for r in records])),'fold_std':float(np.std([r['auc'] for r in records])),'elapsed_seconds':time.time()-start,'status':'complete','submission_sha256':hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()}; (out/'metrics.json').write_text(json.dumps(m,indent=2)+'\n'); print(json.dumps(m,indent=2))
if __name__=='__main__': main()
