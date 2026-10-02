# Playground Series S6E9 研究档案 / Research Archive

Kaggle **Playground Series S6E9 — Predicting Electric Vehicle Purchases** 的本地实验与复现入口。
比赛已于 2026-09-30 结束；本仓库是纯代码归档，竞赛数据、下载的公开件、模型权重、预测文件、
提交回执与账号凭据全部留在本机并被 Git 忽略。

> ## 想了解整个探索脉络，先读这一份
>
> **[`docs/experiment-log-s6e9.md`](docs/experiment-log-s6e9.md)** — 全程时间线总记录。
> 它把 09-08 到 10-02 的完整过程合在一份文件里：基线搭建 → income/generator 特征发现 →
> 自训练与 OOF 高估复盘 → 09-25/26 两轮专项战役 → 09-30 截止日冲刺 → 最终成绩与复盘，
> 附全部 50 次提交的 ref/public/private 明细，以及每份原始记录的位置索引。
>
> 只想看结果和教训：**[`docs/postmortem-s6e9.md`](docs/postmortem-s6e9.md)**（最终名次、
> public/private 反向机理、可复用的选件清单）。
> 系统化的分类教训（验证方法 / 建模死路 / 公开榜机制 / 流程工程 / 合规）：
> **[`docs/lessons-s6e9.md`](docs/lessons-s6e9.md)**。

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
最终 private 第 213。

### 公私榜教训（全场最贵的一课）

public 第 6 → private 第 213，落差不来自模型变弱，而来自**最后一刻按 public 排名选件**。

| 事实 | 数字 |
| --- | --- |
| 截止日 10 次提交的 public/private 相关 | **r = −0.76**（斜率 −0.79） |
| public 第 1 的 B10（我们勾选的） | public 0.94691 → private 0.94551 |
| public 最后 1 的 lucifer `oof_stack` | public 0.94672 → private **0.94565**（约第 79 名） |
| 选件错误的代价 | 约 **134 个名次** |
| 全部 50 次提交的相关 | r = +0.81，斜率 0.41（远离天花板时 public 仍然传导） |

**为什么会反转。** 9/25 之后 public 前沿（0.9468–0.9470）已经不是独立模型，而是对 public 段反复
探针、拟合的产物（megayak 自审：21 步 probing 在诚实 AUC 上是 `−14.45` 单位）。public 分数是
"文件在 public 行上排序"的确定性函数，所以探针能精确读出 public 行标签，但这些收益只存在于
public 行，private 上会回吐。越贴近天花板，public 排序越是负向信号：public 越高 → 对 public
拟合越多 → private 期望越低。当日 private 极差 `0.94542–0.94565`（23 个 1e-4 单位），远大于
噪声——这是选择问题，不是训练问题。

**下次如何规避（写成流程，不靠临场判断）。**

1. **两池记账**：honest 池（严格外层 OOF、零 public 反馈）与 probe 池（用了 public 反馈/探针件
   rank 混合）从第一天起分开；public 分只在 probe 池内部排序有意义。
2. **先估传导率**：用自己历史提交的 public→private 关系判断当前处于"传导区"还是"探针区"
   （本场远离天花板 r=+0.81，探针区 r=−0.76）。候选 public 分差小于噪声（AUC 场约 `2–5e-4`）时，
   一律按 private 期望排序。
3. **定义 private 期望** ≈ 折内 OOF − public 拟合惩罚；惩罚项由"来源链 + 候选间逐行不一致率"估计。
4. **两槽规则写死**：两池各取 private 期望最大者；同一池时，第二槽选与第一槽逐行最不相关的
   高期望候选。**绝不按 public 排名勾选。**
5. **截止日前预注册选件规则**，最后一小时只执行、不重新决策——防止被最后一次 public 刷新带走。
6. **把探针当"买信息"**：leaderboard 取最高分使失败探针不扣分，但每发都要写清假设、E[max] 与
   配额用途，且绝不让 probe 池占据最终两个槽。
7. 赛后把真实传导率与选件结果回填到下一场的 checklist。

完整推演与证据见 [`docs/lessons-s6e9.md`](docs/lessons-s6e9.md) 第 3、4 节与
[`docs/postmortem-s6e9.md`](docs/postmortem-s6e9.md)。

### 推荐阅读路线

1. **[`docs/experiment-log-s6e9.md`](docs/experiment-log-s6e9.md)** — **全程探索脉络（主入口）**；
2. [`docs/postmortem-s6e9.md`](docs/postmortem-s6e9.md) — 最终结果、public→private 反转的证据与选件教训；
3. [`docs/lessons-s6e9.md`](docs/lessons-s6e9.md) — **分类教训清单**（验证/建模/公开榜/选件/流程/合规）；
4. [`docs/script-index.md`](docs/script-index.md) — 72 个脚本的分类索引（主流程 / 可选模型 / 诊断 / 历史）；
5. [`docs/reproducibility.md`](docs/reproducibility.md) — 数据、折划分、编码、工件与复现契约；
6. [`docs/sources.md`](docs/sources.md) — 公开 notebook/讨论来源与许可状态；
7. [`docs/release-checklist.md`](docs/release-checklist.md) — 公开推送前的发布闸门与合规检查；
8. [`docs/top_trick_strategy_2026-09-10.md`](docs/top_trick_strategy_2026-09-10.md) — 赛中的中文策略结论快照。

### 项目结构（详细）

```text
.
|-- README.md                     # 本文件：成绩、阅读路线、结构说明
|-- LICENSE                       # MIT（仅覆盖原创代码，不覆盖数据/第三方材料）
|-- Makefile                      # verify / compile / test / release-check / baseline 等快捷命令
|-- requirements.txt              # 训练与校验的核心依赖
|-- requirements-optional.txt     # TabM / CTBoost 可选依赖（需 CUDA 版 PyTorch）
|-- requirements-kaggle.txt       # Kaggle CLI（独立的 Python 3.11 环境）
|-- scripts/                      # 全部实验入口（72 个脚本 + __init__.py）
|-- tools/                        # 项目校验、运行汇总、发布检查（3 个）
|-- tests/                        # 13 项离线单测（不联网、不提交）
|-- docs/                         # 全部文档（7 份，见下表）
|-- .github/workflows/ci.yml      # CI：compileall + unittest + verify_project + release check
|
|-- （以下均被 .gitignore 忽略，只存在于本机）
|-- train.csv / test.csv / sample_submission.csv
|-- original/                     # 可选原始来源数据（10,000 行）
|-- artifacts/                    # 模型、OOF/test 预测、回执、诊断工件
|-- intel/                        # 公开情报缓存（下载的 notebook / 讨论 / 数据快照）
|-- docs/internal/                # 54 篇赛中日更笔记（模型进展、审计、社区情报）
|-- offline_experiments_2026-09-09.md
|-- catboost_info/  kernel-metadata.json  s6e9-lb-*.ipynb
`-- （仓库外，本机工作区）../s6e9_public/   # 提交账本、探针、混合文件、监控状态
```

#### `scripts/` 分类

| 类别 | 数量 | 代表文件 | 说明 |
| --- | ---: | --- | --- |
| 主流程 / 维护中 | 15 | `train_encoded.py`、`train_10fold_nested.py`、`train_tabm.py`、`train_ctboost.py`、`train_pairwise.py`、`evaluate_offline.py`、`submit_checked.py`、`audit_distribution.py`、`confirm_margin.py` | 有配置/工件契约的正式入口；`train_formula.py`、`train_conditional.py`、`train_income_te.py`、`train_income_cond.py`、`train_orig_lift.py`、`train_orig_model.py` 是包一层的主流程变体 |
| 候选构建器 | 18 | `build_self_candidate.py`、`build_wave7_tabm_transfer.py`、`build_wave9_seed_transfer.py`、`build_full_te_member.py` | 由已有 OOF/test 预测构建融合候选 |
| 机制诊断 | 5 | `diagnose_bagging_transfer.py`、`diagnose_tabm_transfer.py`、`diagnose_full_te_teststyle*.py` | 折平均不对称、test-style 传导等关键机制实验 |
| 审计脚本 | 3 | `audit_local_progress.py`、`audit_commute_discovery.py`、`audit_quarter_rule_seed2026.py` | 训练流程与规则族审计 |
| 历史模型变体 | 25 | `train_generative_*`、`train_hierarchical_te.py`、`train_support_geometry*.py`、`train_xgb_latent_probit.py` | 赛中探索过的模型/特征路线，保留用于溯源 |
| 历史基线 | 4 | `run_baseline.py`、`run_lgbm_features.py`、`run_gam.py`、`run_rank_logistic.py` | 最早一批基线，非推荐起点 |
| 其他 | 2 | `pilot_softpl.py`、`evaluate_full_mod_combo.py` | 软伪标签 pilot、组合评分辅助脚本 |

所有脚本都用 `Path(__file__).resolve().parents[1]` 定位仓库根目录，因此请从仓库根目录运行；
数据与 `artifacts/` 都相对根目录解析。脚本之间以平铺模块互相 import（如
`from train_encoded import ROOT`），`scripts/` 需保持在 `sys.path` 中——直接运行
`python scripts/xxx.py` 或按 `tool`/`test` 的方式导入均可。

#### `tools/`、`tests/`、`docs/`

| 路径 | 作用 |
| --- | --- |
| `tools/verify_project.py` | 数据 schema/ID、Python 语法、工件契约、凭据模式检查（可 `--skip-data --skip-artifacts`） |
| `tools/summarize_runs.py` | 汇总本地 run 的 OOF/Public 分数与提交回执，可写忽略的 `docs/internal/run-index.md` |
| `tools/check_git_release.py` | 检查 Git index：禁止路径、二进制/超大文件、凭据模式 |
| `tests/test_offline_utils.py` | AUC/配对比较/边界规则的单测 |
| `tests/test_repository_tools.py` | 发布检查器与提交守卫（`--confirm-upload`）的单测 |
| `docs/experiment-log-s6e9.md` | **全程时间线总记录（主入口）** |
| `docs/postmortem-s6e9.md` | 赛后复盘：最终成绩、public/private 反转、选件教训 |
| `docs/lessons-s6e9.md` | **分类教训清单**：验证方法、建模死路、公开榜机制、选件规则、流程工程、合规与凭据 |
| `docs/script-index.md` | 脚本分类索引与各自输出约定 |
| `docs/reproducibility.md` | 数据、折、编码、工件、重跑契约 |
| `docs/sources.md` | 公开来源链接与许可待办 |
| `docs/release-checklist.md` | 公开发布/推送前的闸门 |
| `docs/top_trick_strategy_2026-09-10.md` | 赛中的中文策略结论快照 |

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

公开 notebook/讨论只作为情报参考，不构成复制许可；来源记录见 [`docs/sources.md`](docs/sources.md)。
仓库内的 MIT 许可只覆盖原创代码，不覆盖竞赛数据、第三方代码或产物。
Kaggle 规则要求公开分享的竞赛代码发布在 Kaggle 讨论区/notebook；公开发布或改变仓库可见性前，
请先读 [`docs/release-checklist.md`](docs/release-checklist.md)。

---

## English

Local experiments and reproduction entry points for Kaggle's **Playground Series S6E9 —
Predicting Electric Vehicle Purchases**. The competition closed on 2026-09-30. This is a code-only
archive: competition data, downloaded public artifacts, model checkpoints, predictions, submission
receipts and credentials stay local and are ignored by Git.

> ## Read this first for the full exploration story
>
> **[`docs/experiment-log-s6e9.md`](docs/experiment-log-s6e9.md)** — the complete timeline.
> One file covering 2026-09-08 → 2026-10-02: baselines, the income/generator feature discoveries,
> the self-training and OOF-overestimation postmortem, the two 09-25/26 campaigns, the deadline-day
> sprint, the final result, all 50 submissions with ref/public/private scores, and a map of the
> original local records.
>
> Results and lessons only: **[`docs/postmortem-s6e9.md`](docs/postmortem-s6e9.md)**.
> Categorised lessons (validation / modelling dead ends / leaderboard mechanics / process /
> compliance): **[`docs/lessons-s6e9.md`](docs/lessons-s6e9.md)**.

### Final result

| | |
| --- | --- |
| Public (close) | **0.94691, 6th** |
| Private (final) | **0.94551, 213th** (eight-way tie) |
| Selected final submissions | deadline blend `B10` (ref `56711566`) + honest OOF candidate v5 (ref `56557482`) |
| Best private-robust file held | lucifer `oof_stack` (private 0.94565, ~79th, not selected) |

The honest local pipeline topped out around 0.9464x. On the final day a rank blend of public probe
files reached 6th on the public board before finishing 213th on private.

### Public vs private: the most expensive lesson

6th on public → 213th on private was not a modelling failure; it was a **selection failure** made by
following the public ranking at the last minute.

| Fact | Number |
| --- | --- |
| Public/private correlation across the 10 deadline-day submissions | **r = −0.76** (slope −0.79) |
| Public #1 B10 (the file we selected) | public 0.94691 → private 0.94551 |
| Public last lucifer `oof_stack` | public 0.94672 → private **0.94565** (~79th) |
| Cost of the selection mistake | ~**134 places** |
| Correlation across all 50 submissions | r = +0.81, slope 0.41 (public still transmits away from the ceiling) |

**Why it reverses.** After 2026-09-25 the public frontier (0.9468–0.9470) was no longer independent
modelling; it was produced by repeatedly probing and fitting the public split (megayak's own audit:
21 probing moves worth `−14.45` units of honest AUC). The public score is a deterministic function of
a file's ordering on the public rows, so probes can read public labels precisely — but those gains
exist only on public rows and revert on private rows. Near the ceiling the public ranking becomes a
negative signal: higher public means more public fitting and lower private expectation. The
deadline-day private spread was `0.94542–0.94565` (23 units of 1e-4), far above noise: a selection
problem, not a modelling problem.

**How to avoid it next time (a written process, not a judgement call).**

1. **Keep two pools from day one**: an honest pool (strict outer-fold OOF, zero public feedback) and a
   probe pool (public-feedback / probe-file rank blends). Public scores only order candidates inside
   the probe pool.
2. **Estimate the transfer rate first**: use your own public→private history to tell whether you are
   in the transmitting regime (here r=+0.81 away from the ceiling) or the probe regime (r=−0.76).
   When candidate public gaps are below noise (~`2–5e-4` for AUC), rank by private expectation.
3. **Define private expectation** ≈ within-fold OOF − public-fitting penalty, where the penalty comes
   from the source chain and the per-row disagreement between candidates.
4. **Fix the two-slot rule in advance**: take the max private expectation from each pool; if both are
   from the same pool, choose the least correlated high-expectation candidate as the second slot.
   **Never select by public ranking.**
5. **Pre-register the selection rule before the deadline** and only execute it in the final hour — do
   not re-decide after the last public refresh.
6. **Use probes to buy information**: the leaderboard keeps your maximum, so a failed probe costs no
   score, but every probe needs a written hypothesis, E[max] and quota purpose — and the probe pool
   must never occupy the final two slots.
7. Feed the realised transfer rate and selection outcome back into the next competition's checklist.

Full reasoning and evidence: [`docs/lessons-s6e9.md`](docs/lessons-s6e9.md) sections 3–4 and
[`docs/postmortem-s6e9.md`](docs/postmortem-s6e9.md).

### Reading order

1. **[`docs/experiment-log-s6e9.md`](docs/experiment-log-s6e9.md)** — **full exploration timeline (start here)**;
2. [`docs/postmortem-s6e9.md`](docs/postmortem-s6e9.md) — final result, the public→private reversal, selection lessons;
3. [`docs/lessons-s6e9.md`](docs/lessons-s6e9.md) — **categorised lessons** (validation / modelling / leaderboard / selection / process / compliance);
4. [`docs/script-index.md`](docs/script-index.md) — what each of the 72 scripts does;
5. [`docs/reproducibility.md`](docs/reproducibility.md) — data, folds, encoding and artifact contracts;
6. [`docs/sources.md`](docs/sources.md) — public references and license status;
7. [`docs/release-checklist.md`](docs/release-checklist.md) — the public-release gate;
8. [`docs/top_trick_strategy_2026-09-10.md`](docs/top_trick_strategy_2026-09-10.md) — the in-competition strategy snapshot (Chinese).

### Project structure (detailed)

```text
.
|-- README.md                     # this file: result, reading order, structure
|-- LICENSE                       # MIT for original code only
|-- Makefile                      # verify / compile / test / release-check / baseline shortcuts
|-- requirements.txt              # core training and validation dependencies
|-- requirements-optional.txt     # optional TabM / CTBoost deps (CUDA PyTorch first)
|-- requirements-kaggle.txt       # Kaggle CLI (separate Python 3.11 environment)
|-- scripts/                      # all 72 experiment entry points (+ __init__.py)
|-- tools/                        # project verification, run summary, release checks (3 files)
|-- tests/                        # 13 offline unit tests (no network, no submissions)
|-- docs/                         # 7 documents (table below)
|-- .github/workflows/ci.yml      # CI: compileall + unittest + verify_project + release check
|
|-- (ignored by .gitignore, local only)
|-- train.csv / test.csv / sample_submission.csv
|-- original/                     # optional 10,000-row source dataset
|-- artifacts/                    # models, OOF/test predictions, receipts, diagnostics
|-- intel/                        # cached public intel (notebooks, discussions, data snapshots)
|-- docs/internal/                # 54 daily notes written during the competition
|-- offline_experiments_2026-09-09.md
|-- catboost_info/  kernel-metadata.json  s6e9-lb-*.ipynb
`-- (outside the repo) ../s6e9_public/   # submission ledger, probes, blends, monitor state
```

#### `scripts/` categories

| Category | Count | Examples | Notes |
| --- | ---: | --- | --- |
| Maintained pipeline | 15 | `train_encoded.py`, `train_10fold_nested.py`, `train_tabm.py`, `train_ctboost.py`, `train_pairwise.py`, `evaluate_offline.py`, `submit_checked.py`, `audit_distribution.py`, `confirm_margin.py` | Full config/artifact contracts; `train_formula.py`, `train_conditional.py`, `train_income_te.py`, `train_income_cond.py`, `train_orig_lift.py`, `train_orig_model.py` wrap the canonical runner |
| Candidate builders | 18 | `build_self_candidate.py`, `build_wave7_tabm_transfer.py`, `build_wave9_seed_transfer.py`, `build_full_te_member.py` | Blend existing OOF/test predictions into candidates |
| Mechanism diagnostics | 5 | `diagnose_bagging_transfer.py`, `diagnose_tabm_transfer.py`, `diagnose_full_te_teststyle*.py` | Fold-averaging asymmetry and test-style transfer experiments |
| Audits | 3 | `audit_local_progress.py`, `audit_commute_discovery.py`, `audit_quarter_rule_seed2026.py` | Pipeline and rule-family audits |
| Historical model variants | 25 | `train_generative_*`, `train_hierarchical_te.py`, `train_support_geometry*.py`, `train_xgb_latent_probit.py` | Explored during the competition, kept for provenance |
| Historical baselines | 4 | `run_baseline.py`, `run_lgbm_features.py`, `run_gam.py`, `run_rank_logistic.py` | Earliest baselines, not a starting point |
| Other | 2 | `pilot_softpl.py`, `evaluate_full_mod_combo.py` | Soft pseudo-label pilot and combination-scoring helper |

Every script resolves the repository root as `Path(__file__).resolve().parents[1]`, so run scripts
from the repository root; data and `artifacts/` resolve relative to that root. Scripts import each
other as flat modules inside `scripts/` (for example `from train_encoded import ROOT`), so run them
directly (`python scripts/xxx.py`) or import them with `scripts/` on `sys.path`.

#### `tools/`, `tests/`, `docs/`

| Path | Purpose |
| --- | --- |
| `tools/verify_project.py` | Data schema/IDs, Python syntax, artifact contracts, credential patterns (`--skip-data --skip-artifacts` available) |
| `tools/summarize_runs.py` | Summarize local run OOF/public scores and cached receipts; can write the ignored `docs/internal/run-index.md` |
| `tools/check_git_release.py` | Git-index check for forbidden paths, binary/large files and credential patterns |
| `tests/test_offline_utils.py` | Unit tests for AUC, paired comparisons and edge rules |
| `tests/test_repository_tools.py` | Unit tests for the release checker and the `--confirm-upload` submission guard |
| `docs/experiment-log-s6e9.md` | **Full exploration timeline (main entry point)** |
| `docs/postmortem-s6e9.md` | Final result, public/private reversal, selection lessons |
| `docs/lessons-s6e9.md` | **Categorised lessons**: validation, modelling dead ends, leaderboard mechanics, selection rules, process, compliance |
| `docs/script-index.md` | Script categories and output contracts |
| `docs/reproducibility.md` | Data, folds, encoding, artifacts and re-run contracts |
| `docs/sources.md` | Public references and outstanding license review |
| `docs/release-checklist.md` | Pre-publication gate |
| `docs/top_trick_strategy_2026-09-10.md` | In-competition strategy snapshot (Chinese) |

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
