"""Use the public 10k-row source data as an external supervised prior."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import train_encoded as runner

raw_features = runner.features

def _orig_prior(train, test):
    from xgboost import XGBClassifier
    orig = pd.read_csv(next((runner.ROOT/'original').glob('*.csv')))
    target = orig.Will_Buy_EV.eq('Yes').to_numpy(dtype='int8')
    cols=[c for c in test.columns if c!='id']
    allx=pd.concat([orig[cols], train[cols], test[cols]],ignore_index=True)
    Xall=pd.DataFrame(index=allx.index)
    for c in cols:
        if allx[c].dtype=='object':
            Xall[c]=pd.Categorical(allx[c]).codes
        else: Xall[c]=allx[c].astype('float32')
    xo=Xall.iloc[:len(orig)]; xc=Xall.iloc[len(orig):]
    model=XGBClassifier(device='cuda',tree_method='hist',max_bin=1024,n_estimators=500,
        max_depth=4,learning_rate=.05,min_child_weight=10,subsample=.9,colsample_bytree=.9,
        objective='binary:logistic',eval_metric='logloss',n_jobs=8,random_state=101)
    model.fit(xo,target,verbose=False)
    return model.predict_proba(xc)[:,1].astype('float32')

def features(train,test,original):
    base,keys,mappings=raw_features(train,test,original)
    pred=_orig_prior(train,test)
    base['original_model_prior']=pred
    base['original_model_prior_logit']=np.log(np.clip(pred,1e-6,1-1e-6)/(1-np.clip(pred,1e-6,1-1e-6))).astype('float32')
    return base,keys,mappings

if __name__=='__main__':
    run=sys.argv[sys.argv.index('--run')+1]; out=runner.ROOT/'artifacts'/run; out.mkdir(parents=True,exist_ok=True)
    (out/'feature_config.json').write_text(json.dumps({'external':'public 10k source model prior','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    runner.features=features; runner.main()
