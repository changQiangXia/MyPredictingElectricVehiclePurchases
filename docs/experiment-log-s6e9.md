# S6E9 全程探索记录（时间线总览）

本文把三处分散的过程记录——`offline_experiments_2026-09-09.md`、本地 `docs/internal/` 的每日笔记、
`s6e9_public/` 的战役报告——合并成一条时间线，读完这一份即可把握整个比赛的探索脉络。
最终结果与选件复盘的浓缩版见 [`docs/postmortem-s6e9.md`](postmortem-s6e9.md)。

- 比赛：Kaggle Playground Series S6E9 — Predicting Electric Vehicle Purchases
- 任务：合成表格二分类（是否购买电动车），指标 ROC AUC；train 668,665 行 × 13 特征，test 286,571 行，
  正例率 17.46%；公开区资料估计 public 段约 57,314 行（20%）
- 参赛身份：CHEN Xiang（单人）；总计 50 次提交
- 最终结果：**public 0.94691（收盘第 6 名）→ private 0.94551（第 213 名）**
- 记录跨度：竞赛数据 2026-08-12 落盘；实际探索与提交 **2026-09-08 → 2026-10-02（25 天）**，
  其中 09-13 – 09-24 没有活动记录。活跃记录日：09-08、09-09、09-10、09-11、09-12、09-25、09-26、
  09-30、10-02。
- 时间线：09-08 首批提交 → 09-12 自训练峰值 → 09-25/26 两轮战役 → 09-30 截止日冲刺 → 10-02 赛后归档

## 0. 成绩轨迹速览

| 日期 | 里程碑 | 账号 public | 备注 |
| --- | --- | ---: | --- |
| 09-08 | 首批嵌套编码 GBDT 提交 | 0.94641–0.94642 | 3 次提交建立基线 |
| 09-09 | 离线 wave2–wave5 + 边界修正 | 0.94643 | 第 36 名；首个可复现本地锚点 |
| 09-10 | nina engine 组合 + commute 支持规则 | 0.94645 | 一度进入约第 20 名 |
| 09-11 | 高收入尾部探针（v19 −10% for income≥110k） | 0.94647 | 第 14–24 名区间，最后一条上涨的探针 |
| 09-12 | nextday commute rank 候选 | 0.94647 | 自训路线封顶 |
| 09-25 | 复制他人 public 锚点（jazivxt） | 0.94657 | 第 59 名；+263 名的"纯复制" |
| 09-30 | 三个探针文件的 rank 混合（B10） | **0.94691** | **收盘第 6 名** |
| 10-02 | 最终 private 公布 | 0.94551 | 第 213 名（同分 8 队） |

## 1. 阶段 A：基线与核心特征发现（09-08 – 09-09）

### 1.1 建立评测面

- 固定 stratified 5-fold（seed 42）作为唯一比较面，后来补充 10-fold（seed 42 / 2026）。
- 所有监督编码严格 outer/inner：外层验证折不参与 target encoding；每个候选保存
  `oof.npy`、`test.npy`、`submission.csv`、`metrics.json`、来源哈希。
- 主模型：nested target encoding + XGBoost（后加 LightGBM/CatBoost 对照）。

### 1.2 三个真正有效的发现

1. **Income exact-value lookup。** `Annual_Income_USD` 有约 13,214 个取值，相同 exact value 携带非单调
   购买率差异；`30,000` 是合成器 mode-collapse 的尖峰，`≥170,537` 在训练集 393/393 全为正，
   `31,004–41,970` 区间几乎全为负。树按范围切分难以表达数千个离散值身份，exact-value target rate
   把它变成一列数。这是本题最大的单一特征。
2. **Generator 指纹 / 频率痕迹。** 公开原始数据 10,000 行可由 `RandomState(101)` 精确重建；把比赛数据与
   原始数据按值对齐，可得到 source/competition 频率、lift、novel value、支持密度与间距。频率/lift 无标签、
   可安全 transductive 使用；原始标签只能隔离使用。`Daily_Commute_km == 5.0` 是巨大尖峰，但该列只有
   约 805 个取值，树可直接学习，收益远小于 income。
3. **响应面接近 additive。** GLM 交互、additive-only LightGBM、joint-key residual 与多族 GBDT 一致显示：
   高容量、pair lookup、未正则化原生高基数类别会过拟合，反而挤出细粒度单变量形状。

### 1.3 模型与候选（本地 OOF）

| 模型 | OOF AUC | 判断 |
| --- | ---: | --- |
| canonical XGB nested TE | 0.946056 | 主锚点 |
| probit 公式边距 XGB | 0.946129 | 小幅稳定增益，后期主成员 |
| CTBoost | 0.946023 | 低权重成员 |
| TabM + 分段数值嵌入 | 0.945839 | 弱但低相关，后期关键 diversity |
| XGB `rank:pairwise` | 0.944828（fold 0） | 失败，弃 |
| wave5 conservative（多成员 rank 混合） | 0.946390 | 提交 public 0.94643（ref `56119541`） |

边界后处理（训练集确定性事实，非榜单调参）：收入 `≥170,537` 置 1；收入 `31,004–41,970` 或通勤 `≥83` 置 0。

### 1.4 外部文件的教训（第一次）

- Nina 公开 dataset 的 Model43（public 0.94643）混合后 public 持平；mickey 的 micro-blend 团队分 0.94644
  无法用其三个可下载文件复现（我们得到 0.94642/0.94643 ×4）。
- Ravi 的公开 meta 集成团队分 0.94647，但我们直接复制其文件只拿到 **0.94641**。
- 结论：**外部文件的展示分数不等于我们提交后的分数**；没有 OOF 的外部件只能当候选素材，不能当本地证据。

### 1.5 阶段结论

- 同源 GBDT 的换参/seed/stacking 已饱和；同族 OOF→LB 误差约 `1e-4`，低于此量级的差异不可信。
- 账号 public 达到 0.94643（第 36 名），距离当时 Top 20 cutoff（约 0.94645）只差 2–3 个第五位小数单位。

## 2. 阶段 B：自训练登顶 Top-20（09-10 – 09-12）

### 2.1 纯自训练模型

- 10-fold probit XGBoost：OOF 0.946276（比同折基础 +7.1e-5）。
- 公式残差 TabM：5 折 OOF 0.945955；扩到 10 折并加正则化后 0.946157。
- 通勤支持规则审计：70 个规则的完整族审计后，只有"通勤 0.25km 分箱、支持 ≥20、全为 No"有效
  （208 行，+5.4e-6）；exact-income 同类规则会把 203 个正例误判为 0（−1.17e-3），证明硬规则不能推广。
- 最终自训练候选 `offline_self_probit10_tabm_support_reg20_seedblend30_v1`：OOF **0.946419**，
  提交 ref `56143535`，public **0.94637**。

### 2.2 关键复盘：OOF 增益为什么没有兑现

这是全程最有价值的一次方法论修正：

1. **OOF 单折 vs test 十折平均的不对称。** 机制实验（3 组固定留出 × 2 套十折 × 10 成员）显示：
   单成员时 seed 融合增益约 `+9.6e-5`，成员先各自十折平均后只剩 `+5.5e-6`。OOF 上的"种子互补"
   大部分是折模型抽样噪声，test 端早已平均掉。
2. **pooled OOF 的跨折尺度奖励。** TabM 的 pooled 增益 `+8.25e-5`，折内 rank 增益只有 `+9e-6`；
   用无标签分位数校准回到折内水平。
3. 这次 `−6e-5` 的 Public 回落中，Public 抽样波动（57,314 行 bootstrap）只占约 0.8% 概率，
   主因是上述 OOF 高估。

**从此确立的评估纪律**：候选比较必须同时看 solo OOF、折内 rank AUC、以及"每个成员先完成
固定折平均再比较"的 test-style 诊断。

### 2.3 Test-style 诊断与 wave7/wave9 候选

- TabM transfer（3 组留出、成员先平均）：TabM 权重 50% 时相对 XGBoost **+5.19e-5**，三组同向。
- 据此在已有 public 0.94643 的 wave5 锚点上生成 wave7（TabM 20/25/30%），25% 版 OOF 0.946415。
- wave9 seed transfer（probit seed 2026，权重 5/10/15%）把本地锚点推到 OOF **0.946436**。

### 2.4 公开探针与 generator 逆向（09-11）

- 外部 v19 artifact（OOF +1.37e-5 @15%）提交后 public 0.94640，**被证伪**。
- 由 OOF 子群诊断发现其伤害集中在高收入尾部：`≥110k` 上 −10% 的探针提交 public **0.94647**
  （ref `56166953`，约第 14–24 名）；−5% 版本同分。
- wave12（source-context + income `%1000` 各 0.5%）：本地 +8.4e-6、3 留出组 +1.8e-5，
  提交后 public 0.94645 **持平**（ref `56157907`）。
- full-TE 外部候选两次 0.94646（refs `56160168`/`56160799`）；严格审计发现其外层 holdout 参与
  early stopping，降级为探索性。
- Generator 逆向结论：比赛数据不是原始行 bootstrap，而是按源边际重采样外层变量、重算下游条件；
  有用的对象是 value-level 采样强度（income/commute 生成器状态），不是源行身份。
- 公开区复现：megayak 的 `income % 1000` 指纹 solo AUC ≈0.618（与我们一致）；Mikhail XGB 复现 0.94618；
  Lucifer SmoothKeys 0.94425/0.94429；局部核、支持几何、CTGAN 假样本特征均无增益。

### 2.5 阶段结果

09-12 以 `nextday_v19_commute_quarter_rankbottom` 收尾，public **0.94647 ×2**（refs `56176676`/`56176718`），
但 private 只有 0.94532——这已经是后来"public 与 private 反向"的早期信号。自训路线封顶，
Top-20 cutoff 停在 0.94647–0.94648。

## 3. 阶段 C：两轮专项战役（09-25 – 09-26）

### 3.1 Public-rank 战役（预算 5 发）

思路：public 分数是文件在 public 段排序的确定性函数，因此可以把 leaderboard 当 oracle 读。

| 发 | ref | 动作 | 结果 |
| --- | --- | --- | --- |
| 1 | `56553151` | 复制公开 649 文件目录的榜首（jazivxt，CC-BY-SA-4.0） | **0.94657，+263 名 → 第 59** |
| 2 | `56554186` | block swap 136k–140k ↔ 148k–152k | 0.94656（−1u） |
| 3 | `56554232` | block swap 212k–216k ↔ 224k–228k | 0.94655（−2u） |
| 4 | `56554237` | block swap 123k–126k ↔ 129k–132k | 0.94655（−2u） |
| 5 | `56554530` | 单行 promote（id 790501） | 0.94657（无变化） |

量化结论：3,160 对文件的回归显示"实现分差 = 0.038 × 模型预测分差"，即 public 差异主要由切分噪声
决定；单发探针的 E[max] 上限约 `+0.31u`，0.9466–0.9468 的队伍是 50–70 发探针堆出来的，不是更强模型。
公开目录最高就到 0.94657。

### 3.2 Private 战役（预算 5 发，用 4 发）

方向：用诚实 OOF 链（20-fold probit XGB 6-seed → megayak view D → 40-fold → RealMLP+window），
每一步以 fold 一致性 + public 1:1 传导为验收。

| ref | 候选 | OOF | public | private（事后） |
| --- | --- | ---: | ---: | ---: |
| `56555516` | v2 = 0.655·wave9 + 0.345·20-fold XGB | 0.946464 | 0.94641 | 0.94539 |
| `56556218` | v3 = 0.825·v2 + 0.175·megayak D | 0.946483 | 0.94643 | 0.94542 |
| `56556967` | v4 = 0.825·v3 + 0.175·40-fold XGB | 0.946490 | 0.94643 | 0.94541 |
| `56557482` | v5 = 0.70·v4 + RealMLP/window | **0.946503** | 0.94644 | **0.94542** |

四次提交的 OOF 增益与 public 增量近似 1:1（不同行样本），是全程最强的"真实泛化"证据。

### 3.3 被验证排除的 15 条路

refit-on-all（3 留出全负，−2.7e-5 ~ −1.1e-4）、125 成员 logistic stack（−1.2e-4）、
hard/soft pseudo-label（−1.98e-3 / −1.56e-3）、golem 19 成员库（权重全 0）、LightGBM 20-fold、
12 个本地特征视图、9 个模型家族代表、窗口编码、megayak A/B/C/E/F 视图、generator-fingerprint 公开库
（对齐后 −1e-6）、局部核、支持几何残差、modulo 层级 TE、source-context 训练特征。结论：诚实管线
OOF 上限 ≈0.946503，6 种独立搜索协议一致。

## 4. 阶段 D：截止日冲刺（09-30）

### 4.1 当天新局面

public 前沿已从 0.94657 移到 **0.9468–0.9470**，来源是新的公开"探针文件"：

- `nina2025/fork-of-ps-s6e9`（0.94680，rank 混合）；
- `megayak/s6e9-reading-and-fitting-the-public-split`（11 轮 probing 记录：两份被探针重度拟合的文件
  因拟合了 public 的不同区段而互补）；
- geodesic 0.94679、lucifer 48-engine 0.94679 / oof_stack 0.94672。

当天 10 发配额全部用完（14:21–15:25 UTC），核心是 rank 混合这些互补文件：

| 候选 | 权重 | public | private（事后） |
| --- | --- | ---: | ---: |
| geodesic | — | 0.94679 | 0.94561 |
| nina-fork-v5 | — | 0.94680 | 0.94559 |
| lucifer 48-engine | — | 0.94679 | 0.94561 |
| lucifer `oof_stack` | — | 0.94672 | **0.94565** |
| B5 | nina .80 / mega .10 / rohit .10 | 0.94684 | 0.94560 |
| B2 | nina .70 / mega .30 | 0.94689 | 0.94557 |
| B8 | nina .40 / mega .40 / luci .20 | 0.94690 | 0.94555 |
| B1 | nina .50 / mega .50 | 0.94690 | 0.94551 |
| B7 | nina .30 / mega .70 | 0.94688 | 0.94542 |
| **B10** | **nina .30 / mega .50 / luci .20** | **0.94691（第 6 名）** | 0.94551 |

- 复现/工程：`kaggle kernels output` 在本机 `IncompleteRead`，改用 `kagglehub` 下载；
  CLI 上传卡 0%，改用 host 重写为 `storage.googleapis.com` 的 `submit_via_storage.py`，10 发零失败。
- 最终两槽勾选：B10（ref `56711566`）+ 诚实 v5（ref `56557482`），Kaggle 取较好者。

## 5. 最终结果与复盘（10-02）

| | 结果 |
| --- | --- |
| Public（收盘） | 0.94691，第 6 名 |
| Private（最终） | **0.94551，第 213 名**（同分 8 队，按提交时间；3,000+ 有分队伍） |
| 最终计分提交 | `56711566` B10（private 0.94551）；`56557482` v5（0.94542） |
| 手中 private 最高件 | `56710849` lucifer `oof_stack` **0.94565 ≈ 第 79 名（同分 20 队）**，未勾选 |
| 榜首 / 第 5 | 0.94602（Chris Deotte）/ 0.94581 |

**核心事实：截止日 10 发里 public 与 private 相关系数 −0.76（斜率 −0.79）。** public 第 1 的 B10 是
当日 private 最差之一，public 最末的 lucifer `oof_stack` 反而是 private 第 1。全部 50 发整体
`r = +0.81`（远离天花板时 public 仍传导），反转只发生在贴近天花板的"探针区"。

机理：最后 0.0002 分的 public 增益来自对 public split 的拟合（megayak 自审这些 probing 在诚实 AUC 上
−14.45 单位），private 段会回吐；越是把探针文件混到极限（B1/B7/B10），private 越差。

教训（可复用选件清单见 [`postmortem-s6e9.md`](postmortem-s6e9.md)）：

1. public 分差 `< 5e-4` 时不要按 public 排序选件，要按 "OOF − public 拟合惩罚" 排序；
2. 两槽应对冲两池（honest 池 / probe 池）各自的 private 期望最大者，而不是"public 最高 + OOF 最高"；
3. 给每个候选标注来源链：是否 rank-blend 了探针件、是否用了 public 反馈、OOF 出处；
4. 评估必须 test-style（折平均后比较），否则会系统性高估 seed/尺度类增益。

## 6. 全部 50 次提交记录

| # | UTC 日期 | ref | 候选（description） | public | private |
|---|---|---|---|---:|---:|
| 1 | 2026-09-08 | 56105921 | public01_ours10_rank_v1 | 0.94641 | 0.94538 |
| 2 | 2026-09-08 | 56105981 | public01_ours10_rank_edges_v1 | 0.94642 | 0.94539 |
| 3 | 2026-09-08 | 56106042 | public_power_two_v1 | 0.94642 | 0.94536 |
| 4 | 2026-09-09 | 56119541 | offline_wave5_conservative | 0.94643 | 0.94540 |
| 5 | 2026-09-09 | 56121563 | offline_wave6_nina50_edges | 0.94643 | 0.94538 |
| 6 | 2026-09-09 | 56127396 | ravi public ensemble candidate | 0.94641 | 0.94539 |
| 7 | 2026-09-09 | 56128953 | mickey055_edges | 0.94643 | 0.94539 |
| 8 | 2026-09-09 | 56128990 | mickey055_raw | 0.94642 | 0.94538 |
| 9 | 2026-09-09 | 56129153 | mickey050_raw | 0.94643 | 0.94538 |
| 10 | 2026-09-09 | 56129166 | mickey030_raw | 0.94642 | 0.94538 |
| 11 | 2026-09-10 | 56133795 | offline_nina_engine2_edges | 0.94644 | 0.94540 |
| 12 | 2026-09-10 | 56133920 | offline_anchor75_nina25_engine2_edges | 0.94644 | 0.94540 |
| 13 | 2026-09-10 | 56133974 | offline_anchor50_nina50_engine2_edges | 0.94644 | 0.94541 |
| 14 | 2026-09-10 | 56134208 | offline_nina_engine2_edges_commute75 | 0.94645 | 0.94540 |
| 15 | 2026-09-10 | 56134286 | offline_anchor75_nina25_engine2_edges_commute75 | 0.94644 | 0.94541 |
| 16 | 2026-09-10 | 56135496 | offline_nina_engine2_edges_commute75_79p75 | 0.94645 | 0.94536 |
| 17 | 2026-09-10 | 56135571 | offline_nina_engine2_…_commute_support4 | 0.94645 | 0.94533 |
| 18 | 2026-09-10 | 56143535 | self probit10 + tabm reg20 + seedblend30 | 0.94637 | 0.94534 |
| 19 | 2026-09-10 | 56150362 | offline_wave7_tabm_transfer25_v1 | 0.94641 | 0.94539 |
| 20 | 2026-09-11 | 56155861 | offline_wave10_generative_k8_01_v1 | 0.94641 | 0.94540 |
| 21 | 2026-09-11 | 56157907 | offline_wave12_source_context_modulo005005_v1 | 0.94645 | 0.94533 |
| 22 | 2026-09-11 | 56160168 | offline_full_te_modulo_external_v1 | 0.94646 | 0.94533 |
| 23 | 2026-09-11 | 56160799 | offline_frontier_old094645_full25_mod006_v1 | 0.94646 | 0.94533 |
| 24 | 2026-09-11 | 56164598 | offline_full_te10_wf01_mod006_v1 | 0.94642 | 0.94539 |
| 25 | 2026-09-11 | 56164653 | offline_full_te10_prob_wf01_mod006_v1 | 0.94642 | 0.94538 |
| 26 | 2026-09-11 | 56166048 | legtarrr_v19_blend_w015_rank_v1 | 0.94640 | 0.94540 |
| 27 | 2026-09-11 | 56166411 | offline_publicbest_v19_gate30000_110000_w10_v1 | 0.94646 | 0.94534 |
| 28 | 2026-09-11 | 56166953 | probe_publicbest_v19_high_wm010 | 0.94647 | 0.94533 |
| 29 | 2026-09-11 | 56167230 | probe_publicbest_v19_high_wm005 | 0.94647 | 0.94533 |
| 30 | 2026-09-12 | 56176676 | nextday_v19_commute_quarter_rankbottom | 0.94647 | 0.94532 |
| 31 | 2026-09-12 | 56176718 | nextday_v19_commute_quarter_rankbottom | 0.94647 | 0.94532 |
| 32 | 2026-09-25 | 56553151 | public anchor jazivxt-56510229（复制） | 0.94657 | 0.94542 |
| 33 | 2026-09-25 | 56554186 | P1 swap 136k–140k ↔ 148k–152k | 0.94656 | 0.94541 |
| 34 | 2026-09-25 | 56554232 | P3 swap 212k–216k ↔ 224k–228k | 0.94655 | 0.94542 |
| 35 | 2026-09-25 | 56554237 | P4 swap 123k–126k ↔ 129k–132k | 0.94655 | 0.94542 |
| 36 | 2026-09-25 | 56554530 | P5 单行 promote id790501 | 0.94657 | 0.94541 |
| 37 | 2026-09-25 | 56555516 | private v2 wave9 + 20-fold XGB | 0.94641 | 0.94539 |
| 38 | 2026-09-25 | 56556218 | private v3 + megayak view D | 0.94643 | 0.94542 |
| 39 | 2026-09-25 | 56556967 | private v4 + 40-fold XGB | 0.94643 | 0.94541 |
| 40 | 2026-09-25 | 56557482 | private v5 + RealMLP/window | 0.94644 | 0.94542 |
| 41 | 2026-09-30 | 56709913 | geodesic blend | 0.94679 | 0.94561 |
| 42 | 2026-09-30 | 56710506 | nina-fork-v5 | 0.94680 | 0.94559 |
| 43 | 2026-09-30 | 56710842 | lucifer 48-engine | 0.94679 | 0.94561 |
| 44 | 2026-09-30 | 56710849 | lucifer 48-engine oof_stack | 0.94672 | **0.94565** |
| 45 | 2026-09-30 | 56711429 | blend B2 nina70/mega30 | 0.94689 | 0.94557 |
| 46 | 2026-09-30 | 56711462 | blend B1 nina50/mega50 | 0.94690 | 0.94551 |
| 47 | 2026-09-30 | 56711485 | blend B5 nina80/mega10/rohit10 | 0.94684 | 0.94560 |
| 48 | 2026-09-30 | 56711522 | blend B7 nina30/mega70 | 0.94688 | 0.94542 |
| 49 | 2026-09-30 | 56711540 | blend B8 nina40/mega40/luci20 | 0.94690 | 0.94555 |
| 50 | 2026-09-30 | 56711566 | **blend B10 nina30/mega50/luci20（最终勾选）** | **0.94691** | 0.94551 |

## 7. 概念与文件地图

### 7.1 术语速查

| 概念 | 含义 |
| --- | --- |
| nested / outer-inner TE | 外层验证折不参与编码；外层训练行内部再交叉拟合，防止标签泄漏 |
| test-style 诊断 | 每个成员先完成固定折平均，再做种子/模型比较，模拟 test 端行为 |
| public 探针 | 用一次提交读 public 段某个区段的标签信息（分数是排序的确定性函数） |
| block swap / 单行 promote | 交换两段等长排名块 / 把一行提到顶端；用于测量标签不平衡或单行标签 |
| E[max] | leaderboard 取历史最高分时，一次探针的期望收益；本场上限约 +0.31u（1u = 1e-5） |
| honest 池 / probe 池 | 无 public 反馈的 OOF 候选池 / 依赖 public 拟合的候选池；最终两槽应分别取 private 期望最大者 |

### 7.2 文件地图

| 内容 | 位置 | 是否入库 |
| --- | --- | --- |
| 赛后结果与选件复盘 | `docs/postmortem-s6e9.md` | 是 |
| 本文件（全程时间线） | `docs/experiment-log-s6e9.md` | 是 |
| 策略结论（中文） | `docs/top_trick_strategy_2026-09-10.md` | 是 |
| 脚本/复现/来源/发布闸门 | `docs/script-index.md`、`docs/reproducibility.md`、`docs/sources.md`、`docs/release-checklist.md` | 是 |
| 09-09 离线实验全记录（334 行） | `offline_experiments_2026-09-09.md`（仓库根目录，被 `.gitignore` 排除） | 否 |
| 09-10 – 09-12 每日笔记（54 篇 + run-index） | `docs/internal/` | 否 |
| 09-25/26/30 战役报告与提交账本 | `/root/autodl-tmp/s6e9_public/*.md`（仓库外） | 否 |
| 探针/混合文件、下载的公开件、OOF/模型 | `s6e9_public/`、`artifacts/`、`intel/` | 否（数据类一律不入库） |

### 7.3 一句话总结

这是一次"本地诚实模型触到 0.94643–0.94647 的天花板，然后靠公开探针文件在截止日冲进 public 第 6、
又在 private 被打回第 213"的比赛：赢在敢于利用 public 榜的确定性做最后冲刺，输在最后一刻用
public 排序而不是 private 期望来选最终两槽——手里的最优 private 文件（0.94565，约第 79 名）
当时并没有被勾选。
