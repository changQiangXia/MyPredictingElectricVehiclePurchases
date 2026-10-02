"""Prepare several review-only submissions for the next Kaggle quota window.

No network calls or uploads.  Candidates are based on the currently scored
high-income-corrected anchor, with explicitly documented rank transforms.
"""
from pathlib import Path
import hashlib, json
import numpy as np, pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'Will_Buy_EV'

def load(path):
    return pd.read_csv(ROOT / path).set_index('id')[TARGET]
def rank(x):
    return rankdata(np.asarray(x), method='average') / len(x)
def write(name, pred, meta):
    out = ROOT/'artifacts'/name; out.mkdir(parents=True, exist_ok=False)
    sample = pd.read_csv(ROOT/'sample_submission.csv')
    sub = sample.copy(); sub[TARGET] = np.asarray(pred)
    assert sub.id.equals(sample.id) and sub[TARGET].between(0,1).all()
    sub.to_csv(out/'submission.csv', index=False)
    meta = dict(meta, run=name, submitted=False, rows=len(sub),
                submission_sha256=hashlib.sha256((out/'submission.csv').read_bytes()).hexdigest())
    (out/'metrics.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(name, meta['submission_sha256'])

def main():
    # This is the best-scored submission family as of 2026-09-11.
    base_path = 'artifacts/probe_publicbest_v19_high_wm010/submission.csv'
    base = load(base_path); br = rank(base.values)
    # Herm view was trained with strict nested TE. It is weaker solo, but its
    # local paired OOF gain came from a negative correction, so keep only small
    # pre-specified negative weights as exploratory candidates.
    herm = load('artifacts/herm_view_v1/submission.csv'); hr = rank(herm.values)
    for w in (-0.10, -0.20):
        # normalize after negative correction only for numerical convenience;
        # AUC depends on ordering, and this keeps values in [0,1].
        p = (1-w)*br + w*hr
        p = (p-p.min())/(p.max()-p.min())
        write(f'nextday_highwm010_herm_neg{int(abs(w)*100):02d}', p, {
            'base': base_path, 'member': 'artifacts/herm_view_v1/submission.csv',
            'rank_weight_member': w,
            'rationale': 'independent inc_d4 + inc_bin_600 view; exploratory negative residual',
            'local_oof_evidence': 'anchor+Herm negative blend peaked +8.4e-6 on frozen OOF; test-style risk noted',
        })
    # The legtarrr member had stable fixed-holdout OOF gain, but its prior
    # public score was lower. Keep conservative weights for a possible probe.
    leg = load('artifacts/legtarrr_v19_blend_w015_rank_v1/submission.csv'); lr = rank(leg.values)
    for w in (0.03, 0.05):
        p=(1-w)*br+w*lr; p=(p-p.min())/(p.max()-p.min())
        write(f'nextday_highwm010_legtarrr_pos{int(w*100):02d}', p, {
            'base': base_path, 'member': 'artifacts/legtarrr_v19_blend_w015_rank_v1/submission.csv',
            'rank_weight_member': w,
            'rationale': 'small independent residual from legtarrr v19',
            'local_oof_evidence': 'member fixed3 holdouts positive; previous public 0.94640 makes this low-confidence',
        })
    # Keep the strongest already-created income-window probes in a manifest so
    # tomorrow's selection is quick and does not regenerate files.
    picks=[]
    for d in sorted((ROOT/'artifacts').glob('nextday_highwm010_*')):
        m=d/'metrics.json'; s=d/'submission.csv'
        if m.exists() and s.exists():
            picks.append({'path':str(s.relative_to(ROOT)), 'metrics':json.loads(m.read_text())})
    for d in sorted((ROOT/'artifacts').glob('probe_publicbest_v19_gate110_*')):
        m=d/'metrics.json'; s=d/'submission.csv'
        if m.exists() and s.exists():
            picks.append({'path':str(s.relative_to(ROOT)), 'metrics':json.loads(m.read_text())})
    (ROOT/'artifacts'/'nextday_candidate_manifest.json').write_text(json.dumps({
        'created_utc':'2026-09-11', 'base_scored_public':0.94647,
        'candidates':picks, 'note':'Review only; user approval required before upload.'
    }, indent=2)+'\n')

if __name__=='__main__': main()
