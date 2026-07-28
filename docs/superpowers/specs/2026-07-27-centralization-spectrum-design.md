# 多 Agent 系统中心化程度对性能影响的研究 — 系统设计 Spec

日期：2026-07-27
状态：待用户审阅

---

## 1. 研究问题与定位

**研究问题**：当多 agent LLM 系统是一个逐利经济体（agent 消耗算力预算、会破产、通过议价转包任务）时，系统的中心化程度如何影响其算力效率（accuracy-per-token）、任务表现与经济生态（破产率、财富不平等）？

**与现有工作的差异化（护城河）**：通信拓扑影响多 agent 性能的研究已较拥挤（MacNet 2406.07155 的小世界拓扑与协作 scaling law；2505.23352 的"中等稀疏最优"；GPTSwarm / DyLAN / G-Designer 的可学习拓扑；AgentPrune 的通信剪枝）。这些工作都把 agent 当作信息流图上的节点、把 token 当作事后统计的成本。**本工作把 agent 当作逐利的市场主体，token 是内生的预算/货币**：中心化不只是改变信息流图，而是改变市场结构（需求垄断、信息垄断、定价权、交易渠道）。这是与 2505.23352（最接近的竞品）的核心区分点。

**必引并区分的相关工作**：
- 2505.23352 Understanding Information Propagation Effects of Communication Topologies（最接近的论点竞品）
- MacNet 2406.07155（拓扑扫描 + scaling law）
- G-Designer 2410.11782 / DyLAN 2310.02170（学习最优拓扑）
- MAST 2503.13657（动机：MAS 失败多为结构性协调失败）
- AgentPrune 2410.02506（通信冗余/效率对照）

## 2. 中心化光谱（自变量）

累积式六级，每级在前一级基础上多锁一个维度：

| 级别 | 配置 | 中心化增量 |
|---|---|---|
| L0 | 需求去中心化 | 全自主市场：任何 agent 自行取题、交付、redeem，自由议价转包 |
| L1 | 需求中心化 | 唯一 interface agent 垄断对外通道（取题 + 交付领钱）；控制/任务分配权随之落入 interface |
| L2 | 信息中心化 | + interface 垄断外部信息检索（`retrieve`） |
| L3 | 定价中心化 | + interface 垄断全系统合同定价：agent 间合同价格也由 interface 裁定（`set_price`），全面禁止议价；交易自由保留（做不做、和谁做仍自主） |
| L4 | 通信中心化 | + 星型：普通 agent 只能与 interface 通信/交易（蕴含 L3） |
| L5 | 单体 agent 基线 | 只有一个 agent，无协作 —— 中心化的极限/基线参照点 |

排序依据：L4（星型）实现即蕴含 L3（唯一买家 → 定价权自然归 interface），故定价中心化为较弱一级排前。控制/任务分配中心化自 L1 起被"需求垄断"蕴含，不单列。

**保留后门**：长期记忆归属（全局共享 vs 每 agent 私有）实现为可配置项，本期不做消融。

## 3. 系统架构

```
┌─────────────────────────────────────────────┐
│  Agents（每个 = 一个 LLM tool-use 循环）      │
│  私有短期记忆：FIFO 队列 + 目标栈 + 收件箱视图  │
├─────────────────────────────────────────────┤
│  基础设施层（进程内 Python）                   │
│  ├ Action 注册表（按配置过滤每 agent 可见集）   │
│  ├ 任务板（WORLD 挂出的外部合同池）            │
│  ├ 通信系统（每 agent 一个微信式 chat list）    │
│  ├ 合同/托管系统（内外部合同统一状态机）        │
│  ├ 账本（余额、转账、燃烧、铸造、破产判定）      │
│  ├ 长期记忆存储（默认每 agent 私有，归属可配置） │
│  └ 检索后端（可插拔，默认 Chroma 向量数据库）    │
├─────────────────────────────────────────────┤
│  调度器 + 实验记录器（全量 trace 落 JSONL）     │
└─────────────────────────────────────────────┘
```

**框架选择**：Anthropic Messages API + 自建薄编排层。理由：需要绝对控制 ① 每次 action 调用的精确 token 计费（API usage 字段）② 配置间 action 可见性开关 ③ 可复现调度 ④ system prompt 纯净无框架注入。排除 AutoGen/LangGraph 等（其内置编排模式正是本实验的自变量）。排除 Claude Agent SDK（面向单 agent + 层级 subagent，harness 行为污染实验控制）。

**权威状态 vs 私有视图**：基础设施持有世界的权威状态（任务板、聊天记录、合同、账本）；agent 短期记忆是喂进其上下文窗口的私有视图/工作状态。收件箱视图只是通信系统之上的读取游标+摘要投影，FIFO 与目标栈才是真正的私有存储。

## 4. 统一合同模型（核心机制）

外部答题与内部转包用同一套合同动词，无独立 submit_answer：

| | 外部合同（WORLD 挂单） | 内部合同（agent 间转包） |
|---|---|---|
| 发布 | WORLD 把题池挂上任务板，标价 R(q) 公示（按难度分档） | `propose_contract(to, task, price)`，可议价 |
| 接单 | `claim_question(qid)` = 接受 WORLD 合同（独占，其他 agent 不再可见） | `accept_contract(id)` → 从发包方余额托管锁款（余额不足则失败） |
| 交付领钱 | `deliver_work(qid, answer)` → 判分器当场判分 → 支付 R(q) × F1 | `deliver_work(id, content)` → content 送达发包方聊天 + 托管款打给承包方，原子交割 |
| 付款条件 | 按质付款（判分器裁决） | 交付即付（子任务无法机器判分；质量靠重复博弈约束——劣质承包方被市场淘汰，属预期观察的涌现现象） |
| 议价 | 不可议价（WORLD 非 agent，明码标价） | L0–L2 自由议价；L3+ 全面禁止议价，所有合同价格由 interface 裁定（见下） |
| 提交次数 | 每题仅一次交付，交付即关闭（防暴力重试） | — |

- `deliver_work` 只需合同 id：支付对象、金额均从合同登记读出，调用者校验为登记承包方。目标不显式可填是安全设计（消除交错/冒领空间）。
- `cancel_contract`：proposed/unpriced 任一方可取消；accepted 后仅承包方可取消（托管退回发包方），发包方不可单方取消已接合同（防白嫖）。
- `pay(to, amount)` 保留用于合同外自由转账（定金、打赏、救济破产者）；合同结算一律走托管。定价中心化不管制 `pay`（管的是合同价格信号，不是赠与）。

**L3+ 定价中心化下的合同生命周期**（新增 `unpriced` 状态与 `set_price` action）：

```
A: propose_contract(to=B, task)          ← 不带价格（带了也忽略）
       ▼  状态 = unpriced，等待 interface 定价
interface: set_price(contract_id, price) ← 仅 interface 可用（L3+ 才存在）
       ▼  状态 = proposed，等待 B 接受/拒绝
B: accept_contract（锁 A 的托管）→ deliver_work 照旧；或 reject_contract
counter_offer 在 L3+ 对所有人禁用；interface 自己发包时 propose 自带价格（它即定价者）
```

## 5. Action 目录

**计费 action**（"与答题相关"；该回合 LLM 调用的 input+output token 全额扣除）：

| Action | 说明 | 权限门控 |
|---|---|---|
| `retrieve(query)` | 检索语料，返回 top-k 段落 | L2+ 仅 interface |
| `work_on(task_id, thought)` | 对某任务做一步推理，写入私有草稿区 | 所有人 |
| `deliver_work(qid, answer)` **对 WORLD** | 对外交付答案即 redeem（答题相关，计费） | L1+ 仅 interface |

**免费 action**（协调类）：

| Action | 说明 | 权限门控 |
|---|---|---|
| `list_questions()` / `claim_question(qid)` | 看/领任务板 | L1+ 仅 interface |
| `send_message(to, text)` / `read_chat(with)` | 点对点通信 | L4 下普通 agent 的 to 仅限 interface |
| `deliver_work(id, content)` **内部合同** | 交付子任务成果（协调行为，免费），触发托管原子交割 | 所有人 |
| `propose_contract` / `accept_contract` / `reject_contract` / `counter_offer` / `cancel_contract` | 合同生命周期 | L3+ counter_offer 对所有人禁用；L3+ 非 interface 的 propose 不带价格（进入 unpriced）；L4 合同对象仅限 interface |
| `set_price(contract_id, price)` | interface 为 unpriced 合同裁定价格 | 仅 L3+ 且仅 interface |
| `pay(to, amount)` | 自由转账 | L4 仅限与 interface |
| `push_goal(note)` / `pop_goal()` | 维护自己的目标栈 | 所有人 |
| `memory_write(content)` / `memory_search(query)` | 长期记忆存取 | 所有人 |
| `check_balance()` / `list_agents()` | 查余额、通讯录 | 所有人 |

**计费操作化定义**：每 agent 每回合恰好一次 LLM 调用、选一个 action。该回合所选 action 若为计费类，则该次调用 input+output token 全额入账扣除；免费类回合不扣。("答题的思考烧钱，社交谈判免费"的可审计实现。)

## 6. Agent 运行时

每回合上下文构成：

```
┌─ system prompt（固定）────────────────────────┐
│ · 身份 + 同伴名单                              │
│ · 永恒根目标：最大化 token 余额                 │
│ · 本配置下可用 action 与规则说明                │
└──────────────────────────────────────────────┘
┌─ 每回合动态渲染 ───────────────────────────────┐
│ · 当前余额                                     │
│ · 目标栈全量（自底向顶）：                       │
│     [0] 最大化 token 余额  ← 系统压入，不可 pop  │
│     [1..n] agent 自行 push 的子目标（带备注）    │
│ · FIFO 最近 K 条 (action → result)             │
│ · 未读消息摘要                                  │
└──────────────────────────────────────────────┘
```

- 目标栈全量渲染（实践 2–5 层，不截断）；仅 FIFO 滚动淘汰。
- 栈根由系统自动压入且不可 pop；其余层 agent 自维护，备注留痕（预期收益、来源合同 id），兼作决策理由 trace。
- 所有 agent（含 interface）共用同一 system prompt 模板，仅按配置替换"你能做什么"段落。
- **角色 skill（轨迹示范）**：system prompt 末尾附加角色手册——worker 与 interface 各一套 action 轨迹 demo（如 worker：claim→retrieve→work_on→deliver；interface：claim→转包→set_price→对 WORLD 交付），demo 片段按配置裁剪，绝不展示该配置下不可用的 action。注意：skill 增加计费回合的 input 成本，且两角色 skill 长度不同（属角色复杂度的真实成本，六配置内部保持对称）。

## 7. 调度

同步回合制（round-robin）：每轮所有存活 agent 各行动一次；轮内顺序每轮以种子随机打乱（消除先手优势且可复现）。破产 agent 跳过计费行为。回合数即吞吐指标。终止条件：题池清空或达最大轮数。

选回合制而非异步并发：可复现、六配置公平比较、token 计量无竞态。

## 8. 经济系统

- **唯一钱源**：WORLD。新钱只通过"答对题"进入系统（`deliver_work` 判分后支付 R(q) × F1）。内部转包/转账严格守恒；计费燃烧是唯一出口。
- **定价**：R(q) 按难度分档（来源与跳数：HotpotQA 2 跳 < MuSiQue 3 跳 < 4 跳）。难度差异使"评估难度→定价"行为有意义。
- **种子资本 B₀**：每 agent 一次性启动金（解决冷启动：零余额则第一步计费行为都不可行），此后一切净收入仅来自外部奖励。数额 pilot 校准。
- **破产**：余额 ≤ 0 → 冻结计费 action（不能再答题），免费 action 保留（可聊天/收款，理论上可被救济复活——预期观察的涌现现象）。
- **账本守恒律**（核心不变量，做成单元测试）：
  `Σ余额 = Σ种子资本 + Σ WORLD 累计支付 − Σ累计燃烧`
- **公平性**：六配置面对同一题池（同样外部总需求）+ 相同种子资本总额，比谁提取价值多、烧 token 少。

## 9. 配置差分矩阵（实验有效性核心：配置间只差此表）

| 权限 | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| 取题/对 WORLD 交付 | 所有人 | 仅 interface | 仅 interface | 仅 interface | 仅 interface | 单 agent |
| 外部检索 `retrieve` | 所有人 | 所有人 | 仅 interface | 仅 interface | 仅 interface | 单 agent |
| 合同定价权 | 双方议价 | 双方议价 | 双方议价 | interface 裁定（`set_price`），议价全禁 | interface 裁定，议价全禁 | — |
| 通信/交易对象 | 任意 | 任意 | 任意 | 任意 | 仅 interface | — |
| agent 数 | 8 | 1+7 | 1+7 | 1+7 | 1+7 | 1 |

## 10. 任务与数据集

- **主实验：混合难度池 = HotpotQA + MuSiQue**（fullwiki/开放检索设定）。
  - HotpotQA：官方 processed Wikipedia dump（2017-10-01，约 500 万篇首段）；题目由语料构造，gold 段落 100% 在库；每题官方标注 2 gold 段落与 supporting facts（供"检索命中率"分析）。
  - MuSiQue（2–4 跳，防捷径设计）：官方每题配套段落聚合成语料库自建索引（社区标准做法）。
  - 混合池带来真实难度差异 → 定价分档有意义。
  - distractor 设定明确不用（会使 retrieve 失去意义、L2 无物可垄断）。
  - 检索后端：Chroma 向量数据库（默认 embedding：all-MiniLM，本地、无需 GPU/API）。多跳题的第二跳实体不在题面，单次检索（无论稀疏/稠密）常拿不全证据 → 迭代检索/分解/转包有真实价值，是实验设计的 feature。后端可插拔（BM25 稀疏检索留作消融 option）。
- **第二臂：FanOutQA**（广度分解：一题 5–10 篇独立子查询，可并行转包）。与 MuSiQue（深度/串行分解）形成"深度可分解 vs 广度可分解 × 中心化"对比。需单独建索引（较新 Wikipedia dump）。
- **保留 option（实验期拍板）**：GSM8K/MATH/SVAMP（数学）、HumanEval/MBPP（代码）、MMLU（知识问答）、GAIA/WebArena/AssistantBench（agentic 长程）、FRAMES（held-out 评测集）。基础设施任务抽象为 `(题目, 判分器)`，新增 benchmark 边际成本低。数学/单函数代码题协作价值低（cf. 2502.08788），可作"协作增益低"对照组考察"中心化影响是否依赖任务可分解性"。
- 第二期候选（支撑"最优中心化随任务经济结构移动"）：GovSim 2404.16698、AucArena 2310.05746、NegotiationArena 2402.05863。

> **Pilot 语料现状（与上文主实验设定的偏差，必须记录）**：`scripts/prepare_data.py`
> 当前构建的**不是** HotpotQA 官方约 500 万篇 fullwiki dump，而是把每题自带的段落集
> （HotpotQA `distractor` 配置的 gold + distractor 段落，加 MuSiQue 每题配套段落）
> 跨全部抽样题**汇总去重成一个共享语料库**（IRCoT 等工作的常见做法）。gold 段落
> 100% 在库，题目可答，pilot 可跑通。
> - **有效性 caveat**：该语料仅数千段且已按题裁剪，检索显著比 fullwiki 容易 →
>   retrieve 的边际价值被低估，**L2 的信息垄断操纵被削弱**（agent 更容易一次命中，
>   少了迭代检索/转包的压力）。因此 pilot 的 L2 效应量应视为下界，不可直接外推。
> - **主实验升级项**：正式跑前改为 fullwiki dump 建索引（HotpotQA 官方 processed
>   dump + MuSiQue 语料），届时上文 §10 设定与实现一致，L2 操纵才具备完整强度。

## 11. 指标体系（因变量）

- **主指标 accuracy-per-token** = 题池总得分 ÷ 系统消耗 token。分母两个版本：
  - (a) 仅计费 token（经济内效率）
  - (b) 全部 token 含协调回合（真实成本效率）
  - **协调开销 = 免费回合 token / 全部 token**（实现口径，见 `metrics.compute_metrics`）；
    两个 accuracy-per-token 之差可由上述两指标导出，中心化如何改变协调开销是高价值结果。
- **辅助**：总正确率（EM/F1）、清池轮数（吞吐）、破产率与存活比例、期末余额 Gini、转包量与成交价分布、每题检索次数、gold 段落命中率。
- 全量 trace（每回合每 agent 的 prompt/action/result/账变）落 JSONL，离线可重放。

## 12. Baseline 策略

- **主对照 = L0–L5 光谱自身**（六配置同构，仅差 §9 矩阵）。
- 外部锚点（sanity check，不堆框架）：去中心端 = Multi-Agent Debate（2305.14325）；中心端 = AutoGen GroupChat orchestrator（2308.08155）；单体 = L5。

## 13. 实验协议

1. **Pilot**：约 30 题、1 seed、L0+L4+L5 三档。校准：种子资本 B₀、R(q) 定价、最大轮数、FIFO 长度 K 等超参。
2. **主实验**：约 200 题混合池 × 6 配置 × 3 seeds。底模 Haiku 4.5（成本可控、支撑统计显著性）。
3. **模型鲁棒性验证**：差异最大的两档配置用 Sonnet 复跑，确认结论不随模型翻转。
4. 第二臂 FanOutQA 及各保留 option：主实验出结果后拍板。

## 14. 错误处理

- LLM 输出不合法 action → 该回合按免费计，result 返回错误说明供下回合重试；连续错误超限则跳过该回合。
- 计费回合基础设施侧失败（如检索超时）不扣费。
- `accept_contract` 时发包方余额不足 → 失败。
- 冒领交付（非登记承包方调用 `deliver_work`）→ 报错。

## 15. 测试策略

- **单元测试**：账本守恒律；action 权限矩阵（每配置 × 每 action × 每角色）；claim 原子性；托管交割原子性；破产冻结语义。
- **确定性集成测试**：脚本化假 agent（预定义 action 序列，不调 LLM）跑全流程。
- **冒烟测试**：小规模真 LLM 端到端。

## 16. 范围外（本期不做）

- 记忆中心化消融（留配置后门）。
- GovSim/AucArena 等经济博弈任务（第二期）。
- 异步并发调度。
- BM25 稀疏检索消融（默认为 Chroma 向量检索）。
