---
name: failure-analyzer
description: Analyzes failure patterns from memory, diagnoses issues, and generates
  modification plans and planner guidance for strategic decision-making.
metadata:
  version: '1.0'
  category: analysis
  stage:
  - analysis
---

# failure-analyzer

This analysis skill performs comprehensive failure diagnosis and strategic planning.
It reads recent evaluated memory entries from `SkillContext.extra.recent_memory` plus the risk matrix from `SkillContext.extra.memory_matrix`.

It does not read the memory store directly.

Artifacts emitted by this skill include:

- `failure_analysis_report`: structured report with recent outcomes, failure examples, risk-matrix summaries, failure categories, modification plans, selector context, and planner guidance
- `memory_report`: compatibility alias of the combined report
- `analysis_report`: compatibility alias of the combined report
- `memory_summary_report`: backward-compatible recent-memory summary
- `selector_context`: compact hints for search selection
- `meta_skill_context`: evidence and candidate skills for refinement/discovery/combination
