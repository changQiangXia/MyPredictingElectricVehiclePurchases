"""Reproduce an explicitly specified local rank blend and optional support map."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from train_encoded import ROOT, TARGET, digest


def rank(x):
    return rankdata(x, method='average')/len(x)


def commute_zero_map(values, y):
    keys=np.floor(values/.25)*.25
    stats=pd.DataFrame(dict(key=keys,y=y)).groupby('key').y.agg(['size','sum'])
    return stats[(stats['size']>=20)&(stats['sum']==0)]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--components',nargs='+',required=True)
    parser.add_argument('--weights',nargs='+',type=float,required=True)
    parser.add_argument('--commute-support',action='store_true')
    args=parser.parse_args()
    assert len(args.components)==len(args.weights) and all(w>=0 for w in args.weights)
    assert abs(sum(args.weights)-1.)<1e-12
    tr,te=pd.read_csv(ROOT/'train.csv'),pd.read_csv(ROOT/'test.csv')
    y=tr[TARGET].eq('Yes').to_numpy('int8')
    folds=np.load(ROOT/'artifacts/folds_seed42.npy')
    out=ROOT/'artifacts'/args.run
    out.mkdir(parents=True,exist_ok=False)
    predictions=[]
    sources={}
    for array in ('oof','test'):
        parts=[]
        for component,weight in zip(args.components,args.weights):
            path=ROOT/'artifacts'/component/f'{array}.npy'
            p=np.load(path)
            assert p.shape==((len(tr),) if array=='oof' else (len(te),))
            assert np.isfinite(p).all()
            parts.append(weight*rank(p))
            sources.setdefault(component,{})[f'{array}_sha256']=digest(path)
        predictions.append(sum(parts))
    oof,test=predictions
    pre_auc=float(roc_auc_score(y,oof))
    np.save(out/'oof_before_support.npy',oof)
    np.save(out/'test_before_support.npy',test)
    rule_records=[]
    if args.commute_support:
        x=tr.Daily_Commute_km.to_numpy()
        for f in range(5):
            ti,vi=np.flatnonzero(folds!=f),np.flatnonzero(folds==f)
            mapping=commute_zero_map(x[ti],y[ti])
            keys=np.floor(x[vi]/.25)*.25
            mask=np.isin(keys,mapping.index)
            oof[vi[mask]]=0.
            mapping.to_csv(out/f'commute_map_{f}.csv')
            rule_records.append(dict(fold=f,overridden_rows=int(mask.sum()),
                                     contradicted_rows=int(y[vi[mask]].sum())))
        mapping=commute_zero_map(x,y)
        test_keys=np.floor(te.Daily_Commute_km.to_numpy()/.25)*.25
        mask=np.isin(test_keys,mapping.index)
        test[mask]=0.
        mapping.to_csv(out/'commute_map_full.csv')
        rule_records.append(dict(fold='full',overridden_test_rows=int(mask.sum())))
    np.save(out/'oof.npy',oof);np.save(out/'test.npy',test)
    sub=pd.read_csv(ROOT/'sample_submission.csv')
    assert sub.id.equals(te.id) and sub.id.is_unique and ((test>=0)&(test<=1)).all()
    sub[TARGET]=test
    sub.to_csv(out/'submission.csv',index=False)
    reference=np.load(ROOT/'artifacts/offline_wave2_local_neural/oof.npy')
    fd=[]
    for f in range(5):
        m=folds==f
        auc=float(roc_auc_score(y[m],oof[m]))
        base=float(roc_auc_score(y[m],reference[m]))
        fd.append(dict(fold=f,auc=auc,reference_auc=base,delta=auc-base))
    metrics=dict(run=args.run,status='complete',submitted=False,public_auc=None,
        components=dict(zip(args.components,args.weights)),sources=sources,
        oof_auc=float(roc_auc_score(y,oof)),oof_auc_before_support=pre_auc,
        reference='offline_wave2_local_neural',reference_auc=float(roc_auc_score(y,reference)),
        diagnostic_five_fold_groups=fd,commute_support=args.commute_support,rule_records=rule_records,
        selection_note='Exploratory candidate on reused OOF; weights and rule family were selected from local comparisons',
        inherited_rules_note='Historical anchor includes full-label-selected hard edges; scores inherit that limitation',
        script_sha256=digest(Path(__file__)),train_sha256=digest(ROOT/'train.csv'),test_sha256=digest(ROOT/'test.csv'),
        fold_sha256=digest(ROOT/'artifacts/folds_seed42.npy'),oof_sha256=digest(out/'oof.npy'),
        test_prediction_sha256=digest(out/'test.npy'),submission_sha256=digest(out/'submission.csv'))
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (out/'builder.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    main()
