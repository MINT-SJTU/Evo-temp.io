# 公众号推文草稿（Evo-RL 开源发布）

## 标题候选

1. 我们把真机强化学习工作开源了：Evo-RL，跑通从数采到策略迭代的完整闭环
2. 从示教到强化：一个可落地的真机 RL 开源框架 Evo-RL
3. 双臂 SO101 实战复盘：我们如何做真机强化学习并决定开源

## 封面与配图建议

- 封面主视觉：双臂 SO101 工作台 + 标题「Evo-RL Open Source」
- 图 1（占位）：整体流程图（数采 -> Value -> Advantage/Indicator -> Policy）
- 图 2（占位）：项目页 hero demo 截图或短视频二维码
- 图 3（占位）：数据规模实验曲线（含 Full FT/LoRA 对比）
- 图 4（占位）：开源仓库结构图（CLI + modules）

---

## 正文（可直接发）

如果你也在做真机策略学习，你一定很熟悉这个问题：

- 单纯加 demonstration，成功率会提高，但边际收益越来越小；
- 想上强化学习闭环，却常常卡在工程链路不完整，数据、标注、训练和评测很难衔接。

这段时间，我们在 LeRobot 基础上，把一套可执行的真机强化学习流程打磨成了一个开源工程：**Evo-RL**。

它不是“只讲思路”的实验仓库，而是一套从真实机器人数据到策略迭代的闭环工具链。

### 我们开源了什么？

核心是 5 个环节：

1. **人机协同数采（HIL）**
- 支持 teleop + policy 协同录制
- 支持人工接管（intervention）
- 支持 episode 成功/失败标注

2. **数据质量报告**
- 自动统计数据规模、任务覆盖、episode 长度分布
- 输出 success/failure/intervention 比例
- 在训练前先做数据体检

3. **Value Function 训练**
- 从真实轨迹训练 value
- 对后续策略学习提供更强的质量信号

4. **离线 Value/Advantage/Indicator 回写**
- 对每一帧写入 `value / advantage / acp_indicator`
- 让策略训练不再只依赖同质化示教监督

5. **ACP-aware 策略训练**
- 在策略训练阶段消费 indicator 信号
- 形成可迭代的“采集 -> 标注 -> 训练 -> 评测”闭环

### 我们当前的阶段性结果

以双臂 SO101 叠毛巾任务为例（内部阶段结果，后续会持续更新公开基准）：

- D0 数据集规模：300 条 episode，413,134 帧，约 3.82 小时
- 当前观测到：100% 数据的 Full FT 能形成可用 baseline；
- 部分 LoRA 设定在同任务上仍存在明显差距。

这也是我们决定开源的关键原因：
**让更多人能复现这个过程，并一起把“数据规模 vs RL 闭环”的问题做成可比较、可积累的公开基准。**

### 为什么这次开源值得关注？

我们认为真机 RL 的门槛，不只是算法，而是工程闭环。

如果没有统一的数据结构、可追踪标注、可复用训练入口，很多结果都只能停留在“单次实验成功”。

Evo-RL 想做的是把这件事标准化：

- 输入是什么
- 中间标签怎么生成
- 策略如何消费这些标签
- 评测结果如何可复现

### 你可以怎么开始

仓库里给了最小可运行入口：

- `lerobot-human-inloop-record`
- `lerobot-dataset-report`
- `lerobot-value-train`
- `lerobot-value-infer`
- `lerobot-train`（ACP hook）

你可以从单臂任务起步，先跑通一轮闭环，再扩展到双臂和更复杂任务。

### 我们接下来会继续做什么

- 发布更完整的双臂 benchmark 卡片（含实验协议）
- 补齐开源项目页中的公开 demo 与失败案例分析
- 推进在线 RL 阶段的稳定性与安全约束

如果你正在做真机学习，欢迎直接提 issue / PR，或者拿你自己的任务来对齐基准。

我们也非常欢迎两类共建：

- 新机器人后端适配
- 统一评测脚本和可复现实验卡片

---

## 文末引导（CTA）

- 开源仓库：`<REPO_LINK_PLACEHOLDER>`
- 项目主页：`<PROJECT_PAGE_LINK_PLACEHOLDER>`
- 文档入口：`<DOCS_LINK_PLACEHOLDER>`
- 交流方式：`<CONTACT_PLACEHOLDER>`

> 如果你希望我们下一篇详细展开某一块（例如 Value 标注策略、双臂数据协议、或 Full FT vs LoRA 评测范式），欢迎留言，我们按票数优先更新。

---

## 素材占位清单（发文前检查）

- [ ] 替换 `<REPO_LINK_PLACEHOLDER>`
- [ ] 替换 `<PROJECT_PAGE_LINK_PLACEHOLDER>`
- [ ] 替换 `<DOCS_LINK_PLACEHOLDER>`
- [ ] 替换 `<CONTACT_PLACEHOLDER>`
- [ ] 插入 3-5 个 demo GIF / 视频二维码
- [ ] 插入 1 张最新实验曲线图
- [ ] 核对最新成功率数字与日期
