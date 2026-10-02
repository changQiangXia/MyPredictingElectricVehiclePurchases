# S6E9 赛后复盘（2026-10-02 归档）

竞赛：Kaggle Playground Series S6E9 — Predicting Electric Vehicle Purchases。
截止：2026-09-30 23:59 UTC。指标 AUC；private 段按行采样，本场 50 次提交全部拿到了 private 分数。

## 最终成绩

| 项目 | 结果 |
| --- | --- |
| Public Leaderboard（收盘时） | **0.94691 — 第 6 名** |
| Private Leaderboard（最终） | **0.94551 — 第 213 名**（同分 8 队，按提交时间排序；3,000+ 支有分队伍） |
| 实际勾选的最终提交 | `56711566` B10 混合件（public 0.94691 / private 0.94551）+ `56557482` 诚实 v5（0.94644 / 0.94542），Kaggle 取两者较好 |
| Private 榜首 | 0.94602（Chris Deotte）；第 5 名 0.94581 |
| 手中 private 最高件 | `56710849` lucifer `submission_oof_stack.csv`（public 0.94672 / **private 0.94565**，约第 79 名）——没进最终两槽 |

即使按最差的选择结果，最终 private 0.94551 也高于旧 public 锚点 `56105921`/`56553151` 的 private
0.94538/0.94542；但我们离自己手里的最好牌（0.94565，约第 79 名）差了约 134 个名次。

## 截止日 10 次提交：public 与 private 几乎完全反向

| ref | 文件 | public | private |
| --- | --- | --- | --- |
| 56709913 | geodesic-blend | 0.94679 | 0.94561 |
| 56710506 | nina-fork-v5 | 0.94680 | 0.94559 |
| 56710842 | lucifer 48-engine | 0.94679 | 0.94561 |
| 56710849 | lucifer 48-engine `oof_stack` | 0.94672 | **0.94565** |
| 56711429 | B2 nina70/mega30 | 0.94689 | 0.94557 |
| 56711462 | B1 nina50/mega50 | 0.94690 | 0.94551 |
| 56711485 | B5 nina80/mega10/rohit10 | 0.94684 | 0.94560 |
| 56711522 | B7 nina30/mega70 | 0.94688 | 0.94542 |
| 56711540 | B8 nina40/mega40/luci20 | 0.94690 | 0.94555 |
| 56711566 | **B10 nina30/mega50/luci20（最终勾选）** | **0.94691** | 0.94551 |

当日 10 件的 Pearson `r = -0.76`、回归斜率 `-0.79`：**public 第 1 的 B10 是当日 private 倒数
之一，public 最后一名的 lucifer oof_stack 是当日 private 第 1。** 在全部 50 次提交上
`r = +0.81`、斜率 0.41——说明远离分数天花板时 public 仍有传导；但在最后 `0.0002` 分的
"探针区"，public 排序已经翻转成负向信号。

## 机理：为什么在探针区 public 越高、private 越差

1. 9/25 之后 public 前沿（0.9468–0.9470）不再是独立模型，而是对 public split 反复探针、
   拟合的产物（megayak 的公开 kernel 记录 11 轮 probing，其自审测得这些动作在诚实 AUC 上是
   `-14.45` 单位，且最大似然拟合找不到任何 calibration 增益）。
2. 每份探针文件只拟合 public 行的某个区间，所以彼此互补：两个 0.94680/0.94677 的文件
   rank 平均能到 0.94690。但这些拟合收益在 private 行上会回吐。
3. 越把 public 压到极限（B1/B7/B10 把 nina+megayak 两个探针文件混到极限），private 越差；
   保留引擎多样性、少做 public 拟合的件（lucifer 48-engine 的 oof stack、geodesic/nina-fork
   的稳定混合）private 反而稳。
4. 当日候选的 private 极差为 0.94542–0.94565（23 个 1e-4 单位），远大于噪声：这是选择
   问题，不是训练问题。

## 做对的地方

- 当天用"两个互补探针文件的 rank 混合"把 public 从 0.94657 推到 0.94691（第 6 名），
  10/10 配额全部成功利用，没有失手。
- 采用了两槽对冲框架（public 最高 + 保底件），方向正确；问题只在保底槽的 private 期望
  其实低于另一批可选件。
- 工程稳定性：`kaggle kernels output` 在本机 `IncompleteRead` 失败后改用 kagglehub 下载；
  提交走 storage 直传，10 次全部一次成功。

## 做错的地方与教训

1. **选择规则错了。** 在 public 分差 `< 5e-4` 的区间，public 排序不能作为选择依据：
   B10 比 lucifer oof_stack 高 19 单位 public，private 却反向低 14 单位。
2. **没有把"public 拟合度"当作负向特征。** 公开区审计早已给出"探针收益会在 private 反转"
   的结论，第二个槽应该按 private 期望挑（lucifer oof_stack / B5 / geodesic 一类），
   而不是按 OOF 挑诚实 v5。
3. **两槽对冲只做了一半。** 两个槽应取"两池中各自的 private 期望最大者"，而不是
   "public 最大者 + OOF 最大者"。
4. 候选之间的 private 差异主要来自 public 拟合链，而不是模型强度；最终选择必须以
   "来源链"信息为依据。

## 可复用的选件清单（下次同类比赛）

1. 给每份候选标注来源链：是否 rank-blend 了探针件、是否使用了 public 反馈、OOF 出处。
2. 维护两池：honest 池（严格外层 OOF、零 public 反馈）与 probe 池；public 分只对 probe 池
   内部排序有效。
3. 最终两槽取 max(private 期望)，其中 private 期望 ≈ OOF − public 拟合惩罚；若允许两槽，
   第二槽选与第一槽逐行最不相关的高期望候选，而不是 OOF 最高的候选。
4. 用公开榜历史估计传导率；候选 public 分差小于噪声（本场约 2–5e-4）时一律按 private
   期望排序。
5. 记录候选间逐行不一致率；不一致率高的组合在 public 上混合收益大，但也意味着更多
   public 拟合，private 期望应下调。

## 数据与复现

- 50 次提交明细、截止日报告、混合权重表存在本地 `s6e9_public/`（不入库）。
- 训练/复现脚本在 `scripts/` 目录；数据、OOF、模型、下载的公开件与回执均被 `.gitignore` 排除。
- 公开来源与许可状态见 [`docs/sources.md`](sources.md)；发布闸门见
  [`docs/release-checklist.md`](release-checklist.md)。
