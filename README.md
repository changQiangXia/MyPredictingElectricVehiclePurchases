# Playground Series S6E9 研究档案 / Research Archive

Kaggle **Playground Series S6E9 — Predicting Electric Vehicle Purchases** 的本地实验与复现入口。
比赛已于 2026-09-30 结束；本仓库是纯代码归档，竞赛数据、下载的公开件、模型权重、预测文件、
提交回执与账号凭据全部留在本机并被 Git 忽略。

---

## 中文

### 最终成绩

| 项目 | 结果 |
| --- | --- |
| Public（收盘） | **0.94691，第 6 名** |
| Private（最终） | **0.94551，第 213 名**（同分 8 队） |
| 最终勾选提交 | 截止日混合件 `B10`（ref `56711566`）+ 诚实 OOF 候选 v5（ref `56557482`） |
| 手中 private 最高件 | lucifer `oof_stack`（private 0.94565，约第 79 名，未勾选） |

一句话复盘：本地诚实管线在 0.9464x 触顶；截止日靠公开探针文件的 rank 混合冲到 public 第 6，
最终 private 第 213。完整时间线见 [docs/experiment-log-s6e9.md](docs/experiment-log-s6e9.md)，
选件教训见 [docs/postmortem-s6e9.md](docs/postmortem-s6e9.md)。

### 目录结构

```text
.
|-- scripts/          # 全部实验入口（72 个脚本，详见 docs/script-index.md）
|   |-- train_encoded.py        # 主流程：嵌套 target encoding + XGBoost/LightGBM/CatBoost
|   |-- train_10fold_nested.py  # 十折 XGBoost 变体
|   |-- train_tabm.py           # TabM 神经表格模型
|   |-- train_ctboost.py        # CTBoost 多样性模型
|   |-- evaluate_offline.py     # OOF rank 融合与配对 AUC 比较
|   |-- submit_checked.py       # 需显式授权的 Kaggle 提交守卫
|   `-- ...                     # 其余 build_/diagnose_/audit_/wave 历史实验脚本
|-- tools/            # 项目校验、运行汇总、发布检查
|-- tests/            # 13 项离线单测
|-- docs/             # 复现、脚本索引、来源、发布闸门、赛后复盘
|-- train.csv / test.csv / sample_submission.csv   # 本机数据，Git 忽略
`-- artifacts/        # 本机模型/预测/回执，Git 忽略
```

所有脚本用 `Path(__file__).resolve().parents[1]` 定位仓库根目录，因此请从仓库根目录运行，
数据与 `artifacts/` 都相对根目录解析。

### 环境

```bash
python -m pip install -r requirements.txt
# 可选：TabM / CTBoost（需先装 CUDA 版 PyTorch，见 docs/reproducibility.md）
python -m pip install -r requirements-optional.txt
# Kaggle 下载/提交（独立 Python 3.11 环境）
python -m pip install -r requirements-kaggle.txt
```

参考环境：Python 3.10.8 + 单卡 RTX 4090 (24GB)。依赖文件只固定直接依赖，不是完整环境锁。

### 数据

把 `train.csv`、`test.csv`、`sample_submission.csv` 放在仓库根目录（Git 忽略）。
可选的原始来源数据放 `original/`。数据请通过 Kaggle 官方流程获取，并先阅读比赛规则与数据许可。

### 训练与评测

```bash
python scripts/train_encoded.py --model xgb --run repro_xgb_01
python scripts/train_10fold_nested.py --run repro_xgb_10fold_01

# 下面三个需要已存在的 artifacts/folds_seed42.npy（由上面的主流程生成）
python scripts/train_tabm.py --run repro_tabm_01
python scripts/train_ctboost.py --run repro_ctboost_01
python scripts/evaluate_offline.py \
  --run repro_comparison_01 \
  --anchor repro_xgb_01 \
  --models repro_tabm_01 repro_ctboost_01
```

### 常用检查

```bash
python tools/verify_project.py --skip-data --skip-artifacts
python -m compileall -q scripts tools tests
python -m unittest discover -s tests -v
python tools/check_git_release.py
```

`tools/check_git_release.py` 读取 Git index，因此新增/修改的文件需要先 `git add`。
Makefile 提供 `make verify / compile / test / release-check` 以及 `make baseline RUN=<id>` 等快捷方式。

### 提交安全

`scripts/submit_checked.py` 在上传前检查行数、ID 顺序、列名、有限概率、SHA256 与 Kaggle 配额，
并且必须显式传 `--confirm-upload` 才会真正提交。不要把它放进 CI 或 Make 目标。
永远不要把 `KAGGLE_API_TOKEN`、`kaggle.json`、`access_token`、cookie 或任何凭据写进仓库、
脚本、notebook、issue 或提交历史。

### 许可与合规

公开 notebook/讨论只作为情报参考，不构成复制许可；来源记录见 [docs/sources.md](docs/sources.md)。
仓库内的 MIT 许可只覆盖原创代码，不覆盖竞赛数据、第三方代码或产物。
Kaggle 规则要求公开分享的竞赛代码发布在 Kaggle 讨论区/notebook；公开发布或改变仓库可见性前，
请先读 [docs/release-checklist.md](docs/release-checklist.md)。

---

## English

Local experiments and reproduction entry points for Kaggle's **Playground Series S6E9 —
Predicting Electric Vehicle Purchases**. The competition closed on 2026-09-30. This is a code-only
archive: competition data, downloaded public artifacts, model checkpoints, predictions, submission
receipts and credentials stay local and are ignored by Git.

### Final result

| | |
| --- | --- |
| Public (close) | **0.94691, 6th** |
| Private (final) | **0.94551, 213th** (eight-way tie) |
| Selected final submissions | deadline blend `B10` (ref `56711566`) + honest OOF candidate v5 (ref `56557482`) |
| Best private-robust file held | lucifer `oof_stack` (private 0.94565, ~79th, not selected) |

The honest local pipeline topped out around 0.9464x. On the final day a rank blend of public probe
files reached 6th on the public board before finishing 213th on private. Full timeline:
[docs/experiment-log-s6e9.md](docs/experiment-log-s6e9.md); selection lessons:
[docs/postmortem-s6e9.md](docs/postmortem-s6e9.md).

### Layout

```text
.
|-- scripts/          # all experiment entry points (72 scripts; see docs/script-index.md)
|   |-- train_encoded.py        # canonical nested target encoding + XGBoost/LightGBM/CatBoost
|   |-- train_10fold_nested.py  # ten-fold XGBoost variant
|   |-- train_tabm.py           # TabM neural tabular model
|   |-- train_ctboost.py        # CTBoost diversity model
|   |-- evaluate_offline.py     # OOF rank blending and paired AUC comparisons
|   |-- submit_checked.py       # explicitly gated Kaggle submission helper
|   `-- ...                     # remaining build_/diagnose_/audit_/wave experiment scripts
|-- tools/            # project verification, run summaries, release checks
|-- tests/            # 13 offline unit tests
|-- docs/             # reproduction, script index, sources, release gate, postmortem
|-- train.csv / test.csv / sample_submission.csv   # local only, ignored by Git
`-- artifacts/        # local model/prediction/receipt artifacts, ignored by Git
```

Every script resolves the repository root as `Path(__file__).resolve().parents[1]`, so run scripts
from the repository root; data and `artifacts/` resolve relative to that root.

### Environment

```bash
python -m pip install -r requirements.txt
# optional: TabM / CTBoost (install a CUDA-enabled PyTorch build first, see docs/reproducibility.md)
python -m pip install -r requirements-optional.txt
# Kaggle downloads/submissions (separate Python 3.11 environment)
python -m pip install -r requirements-kaggle.txt
```

Reference environment: Python 3.10.8 with one RTX 4090 (24 GB). The requirement files pin direct
dependencies only; they are not a complete environment lock.

### Data

Place `train.csv`, `test.csv`, and `sample_submission.csv` in the repository root (ignored by Git).
Optional source data belongs in `original/`. Obtain data through Kaggle's official flow and review the
competition rules and data license first.

### Training and evaluation

```bash
python scripts/train_encoded.py --model xgb --run repro_xgb_01
python scripts/train_10fold_nested.py --run repro_xgb_10fold_01

# these three require an existing artifacts/folds_seed42.npy created by the canonical run
python scripts/train_tabm.py --run repro_tabm_01
python scripts/train_ctboost.py --run repro_ctboost_01
python scripts/evaluate_offline.py \
  --run repro_comparison_01 \
  --anchor repro_xgb_01 \
  --models repro_tabm_01 repro_ctboost_01
```

### Common checks

```bash
python tools/verify_project.py --skip-data --skip-artifacts
python -m compileall -q scripts tools tests
python -m unittest discover -s tests -v
python tools/check_git_release.py
```

`tools/check_git_release.py` reads the Git index, so stage new or edited files before running it.
The Makefile offers `make verify / compile / test / release-check` and `make baseline RUN=<id>`.

### Submission safety

`scripts/submit_checked.py` validates row count, ID order, columns, finite probabilities, SHA256 and
Kaggle quota before uploading, and requires the explicit `--confirm-upload` flag. Never run it from a
generic CI job or Make target. Never put `KAGGLE_API_TOKEN`, `kaggle.json`, `access_token`, cookies or
any other credential in this repository, shell scripts, notebooks, issue comments or commit history.

### License and compliance

Public notebooks and discussions were used as scouting references, not as a license to copy code,
data or artifacts; sources are recorded in [docs/sources.md](docs/sources.md). The included MIT
license covers original project code only, not competition data or third-party material. Kaggle's
rules require publicly shared competition code to be posted on the competition's discussion forum or
notebooks; read [docs/release-checklist.md](docs/release-checklist.md) before publishing or changing
repository visibility.
