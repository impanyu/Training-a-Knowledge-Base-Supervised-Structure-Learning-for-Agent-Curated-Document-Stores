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

## 2. 中心化配置（自变量，v3：单因子设计）

**v3 转向：不再累积加锁，每种中心化独立测试**——每个配置相对全去中心基线 C0 恰好翻转一个开关，效应可干净归因；选择性组合留作后续实验（开关为独立布尔量，机械上支持任意组合）。

| 配置 | 名称 | 唯一翻转的开关 | hub agent |
|---|---|---|---|
| C0 | 全去中心基线 | 无 | 无 |
| C1 | 需求中心化 | hub 垄断领 task + 对 WORLD 交付 | 有 |
| C2 | 解题记忆中心化 | 全员共享解题知识库（共享库属基础设施，无需 hub） | 无 |
| C3 | 定价中心化 | hub 裁定所有合同价（`set_price`，议价禁）；领单交付权人人平等 | 有 |
| C4 | 信贷中心化 | hub 是唯一出借人；其余全部自由 | 有 |
| C5 | 通信中心化 | 星型：只能与 hub 通信/交易/借贷；但人人可领单、价格可谈（对手只有 hub） | 有 |
| C6 | 共识中心化 | 根目标改为**全局总余额最大化**（集体效用函数）；动态区展示全局余额+个人余额；产权/交易规则全部保持去中心 | 无 |
| C7 | 单体基线 | 1 agent，无协作 | — |

信息中心化已删除（v3）：`retrieve` 打在基础设施共享语料上，信息天然中心化供给，无需 agent 层垄断转售。

**保留后门**：自由文本长期记忆的归属（共享 vs 私有）仍为可配置项，本期不消融（解题记忆的共享已由 C2 覆盖）。

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
| 接单 | `claim_task` = 接受 WORLD 合同（独占，其他 agent 不再可见） | `accept_contract(id)` → 从发包方余额托管锁款（余额不足则失败） |
| 交付领钱 | `deliver_work(task, answers_json)` → 逐叶判分 → Σ R(叶)×F1 | `deliver_work(id, content)` → content 送达发包方聊天 + 托管款打给承包方，原子交割 |
| 付款条件 | 按质付款（判分器裁决） | 交付即付（子任务无法机器判分；质量靠重复博弈约束——劣质承包方被市场淘汰，属预期观察的涌现现象） |
| 议价 | 不可议价（WORLD 非 agent，明码标价） | 默认自由议价；C3 禁止议价、价格由 hub 裁定（见下） |
| 提交次数 | 每题仅一次交付，交付即关闭（防暴力重试） | — |

- `deliver_work` 只需合同 id：支付对象、金额均从合同登记读出，调用者校验为登记承包方。目标不显式可填是安全设计（消除交错/冒领空间）。
- `cancel_contract`：proposed/unpriced 任一方可取消；accepted 后仅承包方可取消（托管退回发包方），发包方不可单方取消已接合同（防白嫖）。
- `pay(to, amount)` 保留用于合同外自由转账（定金、打赏、救济破产者）；合同结算一律走托管。定价中心化不管制 `pay`（管的是合同价格信号，不是赠与）。

**C3 定价中心化下的合同生命周期**（新增 `unpriced` 状态与 `set_price` action）：

```
A: propose_contract(to=B, task)          ← 不带价格（带了也忽略）
       ▼  状态 = unpriced，等待 hub 定价
hub: set_price(contract_id, price) ← 仅 hub 可用（C3 才存在）
       ▼  状态 = proposed，等待 B 接受/拒绝
B: accept_contract（锁 A 的托管）→ deliver_work 照旧；或 reject_contract
counter_offer 在 C3 对所有人禁用；hub 自己发包时 propose 自带价格（它即定价者）
```

**信贷机制（v2 新增）**：
- `propose_loan(to=出借人, amount)` → 出借人 `accept_loan(loan_id)` → 本金划转借款人。
- **利率恒为每轮 1%**（系统常数，不参与议价、不属 hub 定价权）。
- 调度器每轮自动划扣利息（借款人余额不足则欠息滚入本金）；`repay_loan(loan_id, amount)` 随时还本。
- 借款人破产 = 出借人坏账（信用风险真实存在，属预期观察的涌现现象）。
- C4（信贷中心化）：出借人只能是 hub。C5 星型下借贷对象因拓扑限制只能是 hub（权利未限，对手唯一）。

**节点绑定合同（v2）**：`propose_contract` 的 task 字段若指认一个 subtask 节点（句子或短 id）→ 合同为节点绑定：承包方 `deliver_work(contract_id, answers_json)` 必须提交覆盖该节点全部叶子的 JSON {qid: answer}，基础设施做覆盖校验（qid 集合齐全才原子交割托管）；答案质量不判分（重复博弈约束）。task 为自由文本 → 自由合同，交付任意文本（信息买卖等涌现交易保留）。

## 5. Action 目录

**计费规则（v2）：全 action 计费**——每回合恰好一次 LLM 调用、选一个 action，该次调用的 input+output token 无条件从余额扣除。原「计费/免费」二分降级为统计标签（解题类 vs 行政类），用于报告行政/解题开销比：

- **解题类**：retrieve、work_on、decompose、deliver_work（对 WORLD 交付 task）
- **行政类**：其余全部（通信、合同、借贷、转账、目标栈、记忆、查询）

| Action | 说明 | 权限门控 |
|---|---|---|
| `list_tasks(offset?)` | 分页列出挂牌 task（短 id + 一句话总结 + 叶数 + 总悬赏，按价排序） | C1 仅 hub |
| `claim_task(task)` | 独占领取一棵 task 树（参数：句子或短 id） | C1 仅 hub |
| `decompose(node)` | 揭示 subtask 的下级子节点（子 subtask 显示一句话总结，叶显示原题）；参数：句子或短 id | 所有人 |
| `retrieve(query)` | 检索语料，top-k 段落 | 所有人（信息中心化已删除） |
| `work_on(node, thought)` | 对某节点做一步推理，写入私有草稿区 | 所有人 |
| `deliver_work(task, answers_json)` 对 WORLD | 打包交付整棵 task：JSON {qid: answer} 覆盖全部叶子 → 逐叶判分 → Σ R(叶)×F1 一次结清；每 task 一次机会 | C1 仅 hub |
| `deliver_work(contract_id, content)` 内部合同 | 交付合同成果，托管原子交割 | 所有人 |
| `send_message` / `read_chat` | 点对点通信 | C5 下普通 agent 仅限 hub |
| `propose_contract` / `accept_contract` / `reject_contract` / `counter_offer` / `cancel_contract` | 合同生命周期 | C3 counter_offer 全禁、非 hub propose 进入 unpriced；C5 合同对象仅限 hub |
| `set_price(contract_id, price)` | hub 裁定合同价 | 仅 C3 且仅 hub |
| `propose_loan(to, amount)` / `accept_loan(loan_id)` / `repay_loan(loan_id, amount)` | 借贷生命周期（利率恒 1%/轮） | C4 出借人仅限 hub；C5 借贷对象仅限 hub |
| `pay(to, amount)` | 自由转账 | C5 仅限与 hub |
| `push_goal` / `pop_goal`、`memory_write` / `memory_search`、`check_balance` / `list_agents` | 目标栈 / 长期记忆 / 查询 | 所有人 |

**寻址约定**：task/subtask 节点同时有短 id（t0001）与唯一一句话总结；action 参数二者皆可（句子做规范化+近似匹配，不唯一时报错并列候选）。

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
- 所有 agent（含 hub）共用同一 system prompt 模板，仅按配置替换"你能做什么"段落。
- **解题长期记忆（Solution Memory，v3 新增）**：每 agent 一个 KV 库，与自由文本记忆（memory_write/search）完全并行独立。**自动写入**（无需主动存）：`decompose` 时记 `subtask → [子节点名]`；交付答案时（WORLD 打包交付含 F1 标注 / 节点绑定合同交付）记 `question → 答案`；**收到**节点绑定交付物的发包方同样自动入库。**读取走独立 action `recall_solutions(name)`**：递归展开记忆中的分解映射，汇集该子树下全部已知答案并报告缺口——已解过的 subtask 一次调用直接拿全套（shortcut）。C2 配置下该库全员共享。skill 内含使用 demo。
- **角色 skill（轨迹示范）**：system prompt 末尾附加角色手册——worker 与 hub 各一套 action 轨迹 demo（如 worker：claim→retrieve→work_on→deliver；hub：claim→转包→set_price→对 WORLD 交付），demo 片段按配置裁剪，绝不展示该配置下不可用的 action。注意：skill 增加计费回合的 input 成本，且两角色 skill 长度不同（属角色复杂度的真实成本，六配置内部保持对称）。

## 7. 调度

同步回合制（round-robin）：每轮所有存活 agent 各行动一次；轮内顺序每轮以种子随机打乱。每轮开始时：1) 过期 claim 自动释放；2) 贷款利息自动划扣（1%/轮）。终止条件：task 池清空、达最大轮数、或全员破产（无收入渠道的终态）。hub_turns_per_round 为配置旋钮（默认 1），留给「单点信息处理能力」独立实验。

选回合制而非异步并发：可复现、六配置公平比较、token 计量无竞态。

## 8. 经济系统

- **唯一钱源**：WORLD。新钱只通过"答对题"进入系统（`deliver_work` 判分后支付 R(q) × F1）。内部转包/转账严格守恒；计费燃烧是唯一出口。
- **定价**：R(q) 按难度分档（来源与跳数：HotpotQA 2 跳 < MuSiQue 3 跳 < 4 跳）。难度差异使"评估难度→定价"行为有意义。**定价原则：R(q) ≈ 1.5 × 该难度档答对一题的平均实际 token 燃烧量**（pilot 实测校准）——WORLD 出价必须留出 ~50% 利润空间，agent 才有利可图、内部转包定价才有意义。现行叶价（pilot 校准）：2跳 18000 / 3跳 30000 / 4跳 45000；task 价 = Σ 叶价。全 action 计费后需 v2 pilot 重校。
- **种子资本 B₀**：每 agent 一次性启动金（解决冷启动：零余额则第一步计费行为都不可行），此后一切净收入仅来自外部奖励。数额 pilot 校准。
- **破产**：余额 ≤ 0 → 冻结解题类 action；行政类保留（可聊天/收款/借钱翻身——信贷让破产可逆，v2 关键涌现观察点）。
- **信贷**：借贷是余额的内部转移（本金与利息均在 agent 间流转，守恒律不变）；坏账 = 出借人损失。
- **账本守恒律**（核心不变量，做成单元测试）：
  `Σ余额 = Σ种子资本 + Σ WORLD 累计支付 − Σ累计燃烧`
- **公平性**：六配置面对同一题池（同样外部总需求）+ 相同种子资本总额，比谁提取价值多、烧 token 少。

## 9. 配置差分矩阵（v3 单因子；实验有效性核心：配置间只差一格）

| 开关 | C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 |
|---|---|---|---|---|---|---|---|
| 领 task/对 WORLD 交付 | 所有人 | **仅 iface** | 所有人 | 所有人 | 所有人 | 所有人 | 所有人 | 单 agent |
| 解题记忆 | 私有 | 私有 | **全员共享** | 私有 | 私有 | 私有 | 私有 | 私有 |
| 合同定价 | 议价 | 议价 | 议价 | **iface 裁定** | 议价 | 议价* | 议价 | — |
| 借贷出借人 | 任意 | 任意 | 任意 | 任意 | **仅 iface** | 任意* | 任意 | — |
| 通信/交易对象 | 任意 | 任意 | 任意 | 任意 | 任意 | **仅 iface** | 任意 | — |
| 根目标 | 个人余额 | 个人 | 个人 | 个人 | 个人 | 个人 | **全局总余额** | 个人 |
| agent 数 | 8 | 1+7 | 8 | 1+7 | 1+7 | 1+7 | 8 | 1 |

*C5 星型下价格仍可谈、借贷仍自由，只是唯一可能的交易对手是 hub（拓扑限制而非权利限制）。

## 10. 任务与数据集（v2：层级任务树）

**问题池**：120 题（90 HotpotQA + 30 MuSiQue，validation 抽样）。语料 = 配套段落池化去重 → Chroma 向量索引（fullwiki 升级项与 validity caveat 同 v1）。

**共享 subtask 库（自底向上构建）**：
1. 全部题面 embedding（与检索同一 all-MiniLM 模型）；
2. 凝聚式层级聚类：题目 →（每组 ≤3 题）基层 subtask →（≤3 个）中层 subtask →（≤3 个）顶层 subtask；
3. **每个 subtask 由 LLM（gpt-5-mini）在建库时生成唯一一句话总结**（一次性预处理）；句子即身份，同时配短 id（t0001）；
4. **语义局部性不变量**（硬保证，建库时校验）：任意内部节点，子树内部平均 embedding 相似度 > 与兄弟子树间相似度——语义越近的 subtask 必在越低层同一子树汇合；
5. subtask 库全局共享：**task = 库中某内部节点及其子树**，天然嵌套复用（小 task 是大 task 的子问题）；同一 question/subtask 可出现于多个挂牌 task。

**挂牌**：从库中抽 ~30 个节点挂上任务板（深度 2–4、大小混合；树深 ≤4 含根、每节点 ≤3 子 → 单 task ≤27 叶）。task 价 = Σ 叶价，一句话总结即牌面。

**交付与判分**：claim 整棵 task（独占，L1+ 仅 hub）→ `decompose` 逐层揭示 → 打包交付 JSON {qid: answer} 覆盖全部叶 → 逐叶 F1 判分 → Σ R(叶)×F1 一次结清；每 task 一次交付机会。**按 (task, 叶) 重复计价**：同题跨 task 各付各的（专业化红利：第二次遇同类题成本趋零）。

**专业化考察**：叶子/子树跨 task 复用 + 语义局部性 → 可测量 agent 是否演化为特定语义子树的专家（专业化指数：按子树聚类的答题集中度 Herfindahl）。

**保留 option（不变）**：FanOutQA、GSM8K/HumanEval/MMLU、GAIA 类、FRAMES；二期：GovSim/AucArena/NegotiationArena。

## 11. 指标体系（因变量）

- **主指标 accuracy-per-token** = 题池总得分 ÷ 系统消耗 token。分母两个版本：
  - (a) 仅计费 token（经济内效率）
  - (b) 全部 token 含协调回合（真实成本效率）
  - **协调开销 = 免费回合 token / 全部 token**（实现口径，见 `metrics.compute_metrics`）；
    两个 accuracy-per-token 之差可由上述两指标导出，中心化如何改变协调开销是高价值结果。
- **辅助**：总正确率（EM/F1）、清池 turns 数（吞吐）、破产率与存活比例、期末余额 Gini、转包量与成交价分布、每题检索次数、gold 段落命中率。
- **v2 新增**：行政/解题开销比（全计费下按统计标签拆分）、**专业化指数**（每 agent 答题的语义子树集中度 Herfindahl）、task 完成率与深度穿透率、信贷指标（贷款量、利息流、坏账率、破产后借贷翻身次数）。
- 全量 trace（每回合每 agent 的 prompt/action/result/账变）落 JSONL，离线可重放。

## 12. Baseline 策略

- **主对照 = L0–L5 光谱自身**（六配置同构，仅差 §9 矩阵）。
- 外部锚点（sanity check，不堆框架）：去中心端 = Multi-Agent Debate（2305.14325）；中心端 = AutoGen GroupChat orchestrator（2308.08155）；单体 = L5。

## 13. 实验协议

1. **Pilot**：约 30 题、1 seed、L0+L4+L5 三档。校准：种子资本 B₀、R(q) 定价、最大轮数、FIFO 长度 K 等超参。
2. **主实验**：约 200 题混合池 × 6 配置 × 3 seeds。底模 **`gpt-5-mini`**（pilot 7 定版：同题池 L5 单体 gpt-5-mini 29/30 清题无破产，deepseek-v4-flash 仅 6/30 且破产——经济实验要求 agent 具备基本答题与决断能力，弱模型下所有组织结构一同失灵，光谱无从比较）。
3. **可选消融（暂缓）**：`deepseek-v4-flash` / `claude-haiku-4-5` 保留为"组织收益的模型能力门槛"消融臂——pilot 已给出弱模型全线失灵的初步证据。模型按前缀路由（deepseek-*/gpt-*/claude-*），`--model` CLI 注入。
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
