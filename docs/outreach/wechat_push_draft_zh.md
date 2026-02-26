# 公众号推文终稿（卖点强化 + 社区号召版）

## 标题候选

1. 我们正式开源：LeRobot101 上持续推进的真机强化学习项目 Evo-RL
2. 不做一次性 Demo：Evo-RL 在 LeRobot101 持续发布，AgileX PiPER 即将加入
3. 从“跑通一次”到“持续迭代”：Evo-RL 开源真机 RL 全链路
4. 我们想做的，不是一条结果曲线，而是一个长期进化的真机 RL 社区

## 封面建议

- 主标题：`Evo-RL 开源`
- 副标题：`LeRobot101 持续发布中 · AgileX PiPER Coming Soon · 社区任务上传计划`

---

## 正文（可直接发布）

真机强化学习不缺“跑通一次”的截图，缺的是任何团队都能复现并持续迭代的公开工程。

今天我们开源的 **Evo-RL**，不是一次性的实验仓库，而是一套持续推进的真机 RL 工程体系：
从人机协同采集、数据报告，到 Value 训练与优势指示标签训练的真机回环，先在 **LeRobot101（SO101）** 持续发布，再扩展到 **AgileX PiPER（coming soon）**。

### 我们最核心的卖点

1. **持续开源，不是一锤子买卖**
我们会持续发布模型、算法和数据，不做“只发一版就停”的项目。

2. **平台扩展明确**
当前主线是 LeRobot101（SO101），AgileX PiPER 是下一站，并采用同一条可复现链路。

3. **社区是主目标，不是附属功能**
我们会逐步开放任务上传，让更多团队把基于 LeRobot101 和 AgileX PiPER 的任务提交上来，进入可比较、可迭代的公开生态。

### 为什么这件事重要

真机 RL 的真实门槛，不只是算法，而是完整工程闭环：

- 数据如何采、如何标、如何回写；
- 训练如何复现，失败如何定位；
- 不同团队如何在同一协议下横向比较。

Evo-RL 现在已经打通的闭环是：

`HIL采集 -> 数据质量报告 -> Value训练 -> Advantage/Indicator回写 -> 指标条件策略训练 -> 真机再迭代`

这意味着我们的每次实验，都可以沉淀为可复用的公开资产。

### 你现在能直接使用什么

当前版本已开放核心 CLI：

- `lerobot-human-inloop-record`
- `lerobot-dataset-report`
- `lerobot-value-train`
- `lerobot-value-infer`
- `lerobot-train`（指标条件训练路径）

对外用户可按 README 直接跑通一轮最小闭环。

### 我们接下来会持续发布什么

- LeRobot101 的持续迭代模型与数据版本
- AgileX PiPER 任务适配与训练配方（https://www.agilex.ai/）
- 社区任务上传入口与公开 benchmark 卡片

### 社区共建邀请

如果你正在做 LeRobot101 或 AgileX PiPER 任务，欢迎参与：

- 提交任务定义与成功标准
- 提交数据与完整复现命令
- 提交结果表和短视频，参与公开对比

我们希望让真机 RL 从“单点突破”变成“协同迭代”。

---

## 文末 CTA（建议）

如果你在做 LeRobot101 或 AgileX PiPER 任务，欢迎提交任务、数据和复现命令。我们会按统一协议公开评测并持续迭代，让每次实验都沉淀为可复用资产。

- GitHub：`https://github.com/Elvin-yk/evo-lerobot`
- 项目主页：`https://elvin-yk.github.io/evo-lerobot/`
- 文档入口：`https://github.com/Elvin-yk/evo-lerobot/blob/kye/main/README.md`
- 任务投稿入口（占位）：`<TASK_SUBMISSION_LINK_COMING_SOON>`

---

## 配图建议（按段落插图）

1. 三大卖点图卡：`持续开源 / 平台扩展 / 社区共建`
2. 闭环流程图：`HIL -> Report -> Value -> Indicator -> Rollout`
3. 命令卡片图：5 个核心 CLI
4. 任务投稿示意图：`LeRobot101 now + AgileX PiPER soon`

---

## 口径备注（内部）

- “首个”建议统一表述：
  - `在我们当前调研范围内，Evo-RL 是 LeRobot 生态中首个持续推进 LeRobot101 真机 RL 开源的项目。`
- 该口径在官网、README、推文保持一致，避免对外描述不统一。
