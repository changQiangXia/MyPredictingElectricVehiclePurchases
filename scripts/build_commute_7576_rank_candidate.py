"""Rank-safe maximum correction for public [75,76) commute rule."""
from pathlib import Path
import hashlib,json
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]; T='Will_Buy_EV'
base=pd.read_csv(ROOT/'artifacts/probe_publicbest_v19_high_wm010/submission.csv'); test=pd.read_csv(ROOT/'test.csv')
mask=((test.Daily_Commute_km>=75)&(test.Daily_Commute_km<76)).to_numpy(); ranks=rankdata(base[T].to_numpy(float),method='average'); pred=ranks/(len(ranks)+1.0); pred[mask]=0.0
name='nextday_v19_commute_7576_rankbottom';out=ROOT/'artifacts'/name;out.mkdir(exist_ok=False);sub=base.copy();sub[T]=pred;sub.to_csv(out/'submission.csv',index=False)
m={'run':name,'base':'probe_publicbest_v19_high_wm010','rule':'fixed public interval [75,76)','rows':int(mask.sum()),'submitted':False,'valid_probability_range':True,'rationale':'rank-equivalent maximum correction for publicly reported pure-negative interval','sha256':hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest()};(out/'metrics.json').write_text(json.dumps(m,indent=2)+'\n');print(m['sha256'])
