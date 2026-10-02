from pathlib import Path
import json,joblib
import numpy as np,pandas as pd
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier
from train_encoded import ROOT,TARGET,features,digest
OUT=ROOT/'artifacts/experimental/full_te_member_v1';OUT.mkdir(parents=True,exist_ok=True)
tr,te=pd.read_csv(ROOT/'train.csv'),pd.read_csv(ROOT/'test.csv');y=tr[TARGET].eq('Yes').to_numpy('int8'); base,keys,_=features(tr,te,False);n=len(tr); X=base.to_numpy()[:n];Xt=base.to_numpy()[n:];K=keys.to_numpy()[:n];Kt=keys.to_numpy()[n:]
partsA=[X];partsT=[Xt]; encs=[]
for smooth in ('auto',10.,100.):
 e=TargetEncoder(target_type='binary',smooth=smooth,cv=5,shuffle=True,random_state=42);partsA.append(e.fit_transform(K,y).astype('float32'));partsT.append(e.transform(Kt).astype('float32'));encs.append(e)
A=np.column_stack(partsA);T=np.column_stack(partsT)
# median stopping point from full-te diagnostic folds
iters=850
m=XGBClassifier(device='cuda',tree_method='hist',max_bin=1024,n_estimators=iters,max_depth=5,learning_rate=.03,min_child_weight=10,subsample=.9,colsample_bytree=.85,reg_alpha=.071,reg_lambda=2.,objective='binary:logistic',eval_metric='auc',n_jobs=8,random_state=42)
m.fit(A,y,verbose=False);m.set_params(device='cpu');p=m.predict_proba(T)[:,1]
np.save(OUT/'test.npy',p);joblib.dump(encs,OUT/'encoders.joblib');m.save_model(OUT/'model.ubj');pd.DataFrame({'id':te.id,TARGET:p}).to_csv(OUT/'submission.csv',index=False)
(OUT/'metrics.json').write_text(json.dumps({'run':'full_te_member_v1','status':'local_candidate','submitted':False,'n_estimators':iters,'test_sha256':digest(OUT/'submission.csv'),'source_train_sha256':digest(ROOT/'train.csv')},indent=2)+'\n')
print('done',p.min(),p.max())
