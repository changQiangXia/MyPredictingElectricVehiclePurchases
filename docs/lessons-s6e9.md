# S6E9 分类教训清单（Lessons Learned）

本文把整场比赛复盘出的教训按主题整理，每条都给出赛场证据数字。结果与选件主复盘见
[`docs/postmortem-s6e9.md`](postmortem-s6e9.md)，完整时间线见
[`docs/experiment-log-s6e9.md`](experiment-log-s6e9.md)。

## 1. 验证与评估方法

1. **OOF 单折预测与 test 折平均的不对称，会系统性高估融合增益。**
   - 证据：60 个模型的机制实验（3 个固定留出组 × 2 套十折 × 10 个成员）：每个种子只取一个成员时，
     seed 融合增益约 `+9.6e-5`；成员先各自十折平均后再融合，只剩 `+5.5e-6`。
   - 实战后果：自训练候选 OOF 从 0.946391 升到 0.946419，提交后 public 反而从 0.94643 落到 0.94637。
   - 规则：任何 seed/模型融合都要先做"每个成员完成固定折平均"的 test-style 比较；只有 test-style
     仍为正的成员才值得给权重。
2. **pooled OOF 会奖励跨折的分数尺度差异。**
   - 证据：正则化 TabM 的 pooled OOF 增益约 `+8.25e-5`，折算到折内 rank 只有约 `+9.0e-6`；
     用无标签分位数校准后又回到折内水平。
   - 规则：模型比较同时报告三种口径——pooled AUC、折内 rank AUC、test-style 折平均。
3. **低于噪声的 OOF 差异不可行动。**
   - 证据：本场同族 GBDT 的 OOF→Public 残差约 `1e-4`；独立社区账本给出同样的门槛；
     我们多次 `+1e-5` 级改动在 public 上完全无感。
   - 规则：`< 2e-5` 的变化不提交；候选需同时满足 `≥ 2e-5`、`≥ 4/5` 折同向、test-style 为正。
4. **用验证标签发现规则会制造假增益。**
   - 证据：在全量 OOF 上把"纯负 income 组"硬置零可以假造 `+3e-4`；把整个发现流程放进外层折后，
     有效的通勤规则只剩 `+5.4e-6`，而 exact-income 规则族把 203 个正例误判为 0（`−1.17e-3`）。
   - 规则：规则族必须把"发现 + 应用"整体放进外层折；换随机折种子只是稳定性检查，不是独立确认。

## 2. 建模与特征

| 结论 | 证据 |
| --- | --- |
| 主信号是 income exact-value target encoding | 约 13,214 个取值、非单调购买率；`≥170,537` 训练集 393/393 为正 |
| generator 频率/lift 是安全的无标签特征 | 原始 10k 行可由 `RandomState(101)` 精确重建；频率/lift 不需要标签 |
| 响应面接近 additive | 交互 GLM、additive-only LightGBM、多族 GBDT 一致；高容量 pair/cat 特征反而下降 |
| probit 公式边距有小而稳定的增益 | 单模 OOF 0.946129 vs 线性 0.946098 |
| TabM 的价值在低权重 diversity | test-style 三留出组：50% 权重相对 XGBoost `+5.2e-5` |
| income `%1000` 指纹真实但极小 | solo AUC ≈ 0.617–0.618；0.4% 权重只加 `+4.5e-6` |
| 集成已饱和 | 诚实管线 v5 OOF `0.946503` 封顶；6 种独立搜索协议一致在 `±2e-5` 内 |

**已证伪、不要重复的路线**：硬/软伪标签（`−1.6e-3` ~ `−2e-3`）；exact-income 硬规则族；
modulo 残差与 `%1000` 层级 TE；支持几何残差；局部核平滑；CTGAN fakeness 特征；
把原始 10,000 行直接并入训练（`−1e-3` 级）；125 成员 logistic stack（`−1.2e-4`）；
同族 GBDT 的换参/seed 扫描；普通 kNN/重复行/ID 顺序/公式分数。

## 3. 公开榜博弈（AUC + 文件提交的机制）

1. **public 分数是文件在 public 行上排序的确定性函数**：字节级复制别人的文件能精确复现其分数
   （`0.94647 → 0.94657`，+263 名，一次提交零建模）。
2. **leaderboard 取历史最高分**：探针失败不扣分、只花配额；单次探针的期望收益上限约
   `+0.31u`（`1u = 1e-5`），这套方法堆不出大分差。
3. **public 的分数差异大部分是切分噪声**：3,160 对文件回归得到
   `实现分差 = 0.038 × 模型预测分差`；可下载公开文件目录（649 份）最高只到 0.94657。
4. **为 public 拟合的收益会在 private 回吐**：megayak 对 21 步 probing 的自审是
   `−14.45` 单位诚实 AUC；截止日 10 次提交的 public/private 相关系数 `−0.76`。
5. 因此 public 榜可以"读"，但读出的增益必须按 private 期望打折；honest 池与 probe 池
   必须分开记账（见第 4 节）。

## 4. 最终选件（全场最贵的一课）

### 事实

- 截止日 10 次提交的 private 极差 `0.94542–0.94565`（23 个 1e-4 单位）——是**选择问题**，不是训练问题。
- 最终勾选：B10（private 0.94551）+ 诚实 v5（0.94542）→ 最终 0.94551，第 213 名。
- 手中 private 最高件：lucifer `oof_stack`（private 0.94565，约第 79 名）——没有勾选，代价约 134 个名次。
- public 排序在探针区完全反转：public 第 1 的 B10 是当日 private 最差之一；public 最末的
  lucifer `oof_stack` 是当日 private 第 1。

### 可复用选件清单

1. 给每份候选标注**来源链**：是否 rank-blend 了探针件、是否使用了 public 反馈、OOF 出处。
2. 维护两池：**honest 池**（严格外层 OOF、零 public 反馈）与 **probe 池**；public 分只在 probe 池
   内部排序有效。
3. 最终两槽取 `max(private 期望)`，其中 `private 期望 ≈ OOF − public 拟合惩罚`；第二槽优先与
   第一槽逐行最不相关的高期望候选，而不是 OOF 最高者。
4. 用公开榜历史估计传导率；候选 public 分差小于噪声（本场约 `2–5e-4`）时一律按 private 期望排序。
5. 记录候选间逐行不一致率：不一致率高意味着 public 混合收益大，也意味着更多 public 拟合，
   private 期望要下调。
6. Kaggle 若允许两个 final slot，先写死"两池各取 max"规则再勾选，避免最后一刻被 public 排名带走。

## 5. 流程与工程

1. **提交前预注册**：每发提交先写假设、预期收益、衡量对象，提交后回填实测（本地
   `s6e9_public/submission_ledger.md` 对 5 发探针完整执行了这套流程）——防止事后解释。
2. **配额纪律**：始终保留机动槽；本场截止日 10/10 用满，但用途是预先规划的（读公开件 + 混合 + 保底）。
3. **工具要有备用通道**：
   - `kaggle kernels output` 在本机 `IncompleteRead` 失败 → 改用 `kagglehub.notebook_output_download`；
   - Kaggle CLI 上传卡在 0% → host 重写为 `storage.googleapis.com` + 显式超时；
   - GitHub 传输慢 → `source /etc/network_turbo`（AutoDL 学术加速）。
4. **工件纪律**：每个候选保存 OOF/test/每折分数/metrics/SHA256/提交回执；数据与回执不入 Git。
5. **别让同一份文件"身份不明"**：外部文件必须标记作者、许可、下载时间与版本，
   否则最后无法判断它是 probe 还是 honest。

## 6. 合规与凭据

1. Kaggle 规则：公开分享竞赛代码必须发布在 Kaggle 讨论区/notebook；私有分享仅限团队成员；
   数据与数据派生工件不进入公开仓库。
2. 凭据：token 不写入 remote URL、`.git/config`、脚本、notebook、提交信息或对话；
   一旦暴露立即吊销并轮换；优先用 credential helper / `gh auth login` / SSH key。
3. 发布闸门：暂存区扫描 + 全历史检查 + 体积/二进制检查 + 第三方许可核对
   （见 [`docs/release-checklist.md`](release-checklist.md)）。

## 7. 一句话

这次赢在**敢用 public 机制做最后执行**，输在**最后两槽仍按 public 排序选件**；
下次把第 4 节的清单写成流程，比再训一个同源模型值钱得多。

## 8. 证据与出处

| 内容 | 位置 | 是否入库 |
| --- | --- | --- |
| 结果、public/private 反转、选件复盘 | `docs/postmortem-s6e9.md` | 是 |
| 全程时间线（含 50 次提交明细） | `docs/experiment-log-s6e9.md` | 是 |
| 建模结论与无效路线 | `docs/top_trick_strategy_2026-09-10.md` | 是 |
| OOF/test-style 机制实验细节 | 本地 `docs/internal/postmortem_2026-09-11.md` | 否 |
| 探针预注册账本、E[max] 推导 | 本地 `s6e9_public/submission_ledger.md`、`campaign_report.md` | 否 |
| private 搜索与 15 条死路 | 本地 `s6e9_public/private_campaign_report.md` | 否 |
| 截止日 10 发与混合权重 | 本地 `s6e9_public/deadline_day_report.md` | 否 |
| 支持规则族审计、modulo/source-context 实验 | 本地 `docs/internal/self_model_progress_2026-09-10.md`、`model_progress_2026-09-11.md` | 否 |
