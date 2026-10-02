"""Build the small, edge-preserving generative-density transfer candidate."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from train_encoded import ROOT, TARGET, digest


def ranks(x):
    return rankdata(x, method="average") / len(x)


def edges(frame, values):
    result = values.copy()
    result[frame.Annual_Income_USD.ge(170537).to_numpy()] = 1.0
    result[(frame.Annual_Income_USD.between(31004, 41970) |
            frame.Daily_Commute_km.ge(83)).to_numpy()] = 0.0
    return result


def main():
    run = "offline_wave10_generative_k8_01_v1"
    out = ROOT / "artifacts" / run
    out.mkdir(exist_ok=False)
    tr, te = pd.read_csv(ROOT/"train.csv"), pd.read_csv(ROOT/"test.csv")
    y = tr[TARGET].eq("Yes").to_numpy()
    anchor_oof = np.load(ROOT/"artifacts/offline_wave7_tabm_transfer25_v1/oof.npy")
    anchor_test = np.load(ROOT/"artifacts/offline_wave7_tabm_transfer25_v1/test.npy")
    gen_oof = np.load(ROOT/"artifacts/generative_mixture_oof_v1/oof_k8.npy")
    gen_test = np.load(ROOT/"artifacts/generative_mixture_oof_v1/test_k8.npy")
    weight = .01
    blend_oof = (1-weight)*ranks(anchor_oof) + weight*ranks(gen_oof)
    blend_test = (1-weight)*ranks(anchor_test) + weight*ranks(gen_test)
    final_oof = edges(tr, blend_oof)
    final_test = edges(te, blend_test)
    fold_ids = np.load(ROOT/"artifacts/folds_seed42.npy")
    anchor_auc = float(roc_auc_score(y, edges(tr, anchor_oof)))
    rec = dict(run=run, status="local_candidate", submitted=False, public_auc=None,
               anchor="offline_wave7_tabm_transfer25_v1", components={"anchor":.99,"generative_k8":.01},
               oof_auc=float(roc_auc_score(y, final_oof)), anchor_oof_auc=anchor_auc,
               gain=float(roc_auc_score(y, final_oof)-anchor_auc), fold_gains=[],
               rank_correlation=float(np.corrcoef(ranks(anchor_oof), ranks(gen_oof))[0,1]),
               source_oof="artifacts/generative_mixture_oof_v1/oof_k8.npy",
               source_test="artifacts/generative_mixture_oof_v1/test_k8.npy",
               test_style_transfer=json.loads((ROOT/"artifacts/generative_mixture_oof_v1/test_style_transfer.json").read_text()) if (ROOT/"artifacts/generative_mixture_oof_v1/test_style_transfer.json").exists() else None,
               code_sha256=digest(Path(__file__)), submitted_file=None)
    for f in range(5):
        m=fold_ids==f
        rec["fold_gains"].append(float(roc_auc_score(y[m], final_oof[m])-roc_auc_score(y[m], edges(tr,anchor_oof)[m])))
    np.save(out/"oof.npy", final_oof); np.save(out/"test.npy", final_test)
    pd.DataFrame({"id":te.id,TARGET:final_test}).to_csv(out/"submission.csv",index=False)
    rec.update(submission_sha256=digest(out/"submission.csv"), train_sha256=digest(ROOT/"train.csv"), test_sha256=digest(ROOT/"test.csv"), fold_sha256=digest(ROOT/"artifacts/folds_seed42.npy"))
    (out/"metrics.json").write_text(json.dumps(rec,indent=2)+"\n")
    (out/"lineage.json").write_text(json.dumps({"anchor":digest(ROOT/"artifacts/offline_wave7_tabm_transfer25_v1/submission.csv"),"generator_oof":digest(ROOT/"artifacts/generative_mixture_oof_v1/oof_k8.npy"),"generator_test":digest(ROOT/"artifacts/generative_mixture_oof_v1/test_k8.npy"),"submitted":False},indent=2)+"\n")
    print(json.dumps(rec,indent=2))


if __name__ == "__main__": main()
