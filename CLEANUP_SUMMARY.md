# 代码变更记录

---

## 2026-04-23

### 一、移除安全约束（commit 04127ff）

**删除**

- `core/environment.py`：删除 `BLOCKED_MARKERS` 关键词列表及 `local_safety_gate` 拦截逻辑
- `configs/config.yaml`：删除 `constraints` 配置字段
- `core/schemas.py`：删除 `SkillContext.constraints` 字段
- `core/evaluator.py`：删除 `usefulness_score`（该字段始终为 0，无实际用途）
- `core/planner_loop.py`：删除向 skill context 传入 constraints 的代码

**原因**：框架定位为红队攻击测试，安全拦截会过滤掉有效攻击技术的评估结果，与框架目标相悖。

---

### 二、修正评估逻辑 + 清理冗余代码（commit ee0af59）

**修改**

- `core/evaluator.py`：攻击成功判定方向修正——`response_risk_score > 0.65` 为成功（原来是 `< 0.35`）
  -   原因：高风险得分表示模型产出了不安全内容，是红队视角下的成功，原逻辑方向相反

**删除**

- `core/planner_loop.py`：删除 `LEGACY_LLM_BACKEND = "openai_compatible"` 常量、`_llm_config_from()`、`_normalize_backend_name()` 方法
- `core/environment.py`：删除 `if backend == "openai_compatible": backend = "llm"` 转换逻辑
  - 原因：全部配置文件已统一使用 `backend: llm` 格式，兼容旧格式的代码成为死代码

- `configs/workflows/basic.yaml`：删除独立的 `refusal_high`、`repeated_failures` 两个条件定义
  - 原因：两个条件触发的动作完全相同（进入 analysis 阶段），合并为单一 `should_analyze`（OR 逻辑）更清晰

- `tests/test_environment.py`：删除 `test_remote_environment_blocks_unsafe_candidate_locally`
  - 原因：对应的 `local_safety_gate` 功能已删除，测试已过时

**新增**

- `core/workflow.py`：添加复合条件支持（`any` / `all` 聚合），为 `should_analyze` 合并条件提供基础
- `tests/test_workflow.py`：补充复合条件的测试覆盖

---

## 2026-04-24

### 重新设计 Skills 触发与调度流程

**修改**

- `core/planner.py`：重写 `_next_search_target()` 方法
  - 原来：优先执行未尝试过的 skill，再按 `best_recent_skill` 重试
  - 现在：从 memory 的 `risk_matrix` 读取历史 ASR，按 ASR 从高到低依次执行
  - 原因：按效果排序，先执行历史表现好的 skill，提高搜索效率

- `core/planner.py`：修改 `route_after_evaluation()` — 搜索完所有 skills 后自动进入 analysis；修改 `advance_after_action()` — analysis 完成后直接进入 meta
  - 原因：明确阶段边界，search → analysis → meta 线性推进，不再依赖模糊的条件触发


**删除**

- `configs/workflows/basic.yaml`：删除 `discover-skill`（从 meta group 移除）
  - 原因：聚焦于优化现有 skills（refine/combine），不再自动生成新策略

- 大幅简化分析逻辑，`skills/failure-analyzer/scripts/run.py`中删除：
  - `best_skills`（按 ASR 排序）、`weak_skills`（按 needs_refinement 排序）、`high_refusal_skills` 等多套指标计算
    - 原因：多套指标标准不统一，ASR 高的 skills 不一定 response_risk 高，导致推荐与决策相互冲突
  - `failure_categories` 中的 8 种分类（`high_refusal`、`low_success_rate`、`combination_opportunity` 等）
    - 原因：分类复杂但对最终决策贡献有限，简化为只检查 `untried_workflow_skills`
  - per-skill 详细建议（`per_skill` 字段、`target_skills` 列表）
    - 原因：细粒度建议增加复杂度，实际效果未见提升
  - 5 场景决策树（discover / combine / refine / weak_skills fallback）
    - 原因：合并为 3 步决策（untried → combine / refine），逻辑更清晰

**新增**

- `core/planner.py`：新增 `_get_skills_sorted_by_asr()`、`_get_better_skills()` 方法
- `core/planner_loop.py`：新增 `_compute_better_skills()`，在调用 failure-analyzer 前计算并注入 `better_skills`
  - 原因：将指标计算职责从 failure-analyzer 移到 planner_loop，failure-analyzer 只负责接收并做决策，职责分离更清晰

**统一评估指标**：全部 meta 决策改用单一标准 `better_skills`（`response_risk` 高 + `refusal` 低），替代原来多套冲突指标。

---

## 2026-04-27

### 一、替换 analysis skill：memory-summarize → failure-analyzer

**删除**

- `skills/memory-summarize/SKILL.md`、`skills/memory-summarize/references/notes.md`、`skills/memory-summarize/scripts/run.py`
  - 原因：功能由 `failure-analyzer` 完整覆盖，旧 skill 不再使用

---

### 二、评估严格化（core/evaluator.py）

**修改**

- `fallback_to_heuristic` 默认值：`True` → `False`
  - 原因：guard 模型失败时静默降级为启发式得分不可信，会污染 memory 中的历史评估数据，改为直接抛出异常
- 修复 bundle 排序方向：高 `response_risk_score` + 低 `refusal_score` 排前（原来符号写反，最优结果反而排末尾）

**新增**

- candidate 与 response 数量不匹配时抛出 `RuntimeError`（防止错位评估）
- guard enabled 但未返回任何分数时抛出 `RuntimeError`（防止静默空结果）

---

### 三、Workflow 扩展（configs/workflows/basic.yaml）

**修改**

- 启用全部 8 个 attack skills（原来只启用 `rewrite-char` + `rewrite-history`）
  - 新启用：`rewrite-emoji`、`rewrite-hypo`、`rewrite-language`、`rewrite-literary`、`rewrite-reverse`、`rewrite-space`、`rewrite-word`
  - 原因：覆盖更多攻击策略，提高搜索阶段的探索广度
- 连续失败触发阈值：2 → 5
  - 原因：阈值过低导致过早进入 analysis，search 阶段未充分覆盖所有 skills

---

### 四、Planner 调度修正（core/planner.py）

**修改**

- `route_after_evaluation()`：新增 `registry` 参数；META 阶段 evaluate 后回到 ANALYSIS（重新计算 better_skills）；切换回 SEARCH 时重置 `selected_skill_names` 和 `consecutive_failures`
- `advance_after_action()`：读取 failure-analyzer 的 `should_stop` 字段决定是否停止；meta skill 完成后保持在 META 阶段（原来跳回 SEARCH）
  - 原因：meta 阶段需要循环优化，每轮执行后重新分析再决策下一步，不应中途跳回 search
- 预算检查逻辑：将 `budget <= 0` 的检查移到 `pending_candidates` 处理之后
  - 原因：原来检查位置可能打断飞行中的 invoke→execute→evaluate 周期，导致候选数据丢失

**新增**

- `_get_better_skills()`、`_get_skills_sorted_by_asr()`：Planner 侧的 fallback 计算，当 failure-analyzer 未给出有效 action 时使用

---

### 五、Planner Loop 改进（core/planner_loop.py）

**修改**

- 步骤预算计费：主循环改为 `while True`，仅 `invoke_skill`、`analyze_memory`、`invoke_meta_skill` 消耗步骤配额
  - 原因：`execute_candidates`、`evaluate_candidates` 是机械性中间步骤（执行已有候选、评估已有响应），不应占用用户步骤配额
- `invoke_skill` 时改为追加到 `selected_skill_names`（原来每次覆盖整个列表）
  - 原因：累积列表用于判断 search 阶段是否已执行完所有 skills
- 环境调用预算耗尽时：若已收集部分 response，修剪 candidates 列表使两者对齐，完成当前评估周期后再停止
  - 原因：原来直接 STOP 会丢弃已收集的 response 数据

**新增**

- `_STEP_CONSUMING_ACTIONS` 常量：明确列出哪些动作计费
- candidate/response 数量不匹配时抛出 `RuntimeError`（与 evaluator 侧校验对应）
- `_compute_better_skills()`：为 failure-analyzer 调用前注入 `better_skills`

---

### 六、其他修改

- `main.py`：启动时清除全局代理环境变量，设置 `NO_PROXY` 排除内部集群服务
  - 原因：集群内部服务若走代理会导致连接失败
- `core/memory_store.py`：`failure_entries` 筛选条件简化为仅 `not success`，删除 `refusal_score >= 0.7` 和 `response_risk_score >= 0.6` 的额外判断
  - 原因：原条件把部分成功的条目也纳入 failure_entries，语义不清晰
- `core/run_report.py`：recorder 补充记录 `should_invoke_meta_skill` 字段，便于调试阶段转换决策
