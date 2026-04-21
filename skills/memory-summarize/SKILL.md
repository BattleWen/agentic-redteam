---
name: memory-summarize
description: Use during analysis to combine recent-memory
  summarization, failure diagnosis, modification planning, and planner guidance
  into one structured report.
metadata:
  version: '1.0'
  category: analysis
  stage:
  - analysis
---

# memory-summarize

This harmless analysis skill combines the old memory summarization and retrieval-style analysis steps into one report.
It reads recent evaluated memory entries from `SkillContext.extra.recent_memory` plus the risk matrix from `SkillContext.extra.memory_matrix`.

It does not read the memory store directly.

Artifacts emitted by this skill include:

- `failure_analysis_report`: structured report with recent outcomes, failure examples, risk-matrix summaries, failure categories, modification plans, selector context, and planner guidance
- `memory_report`: compatibility alias of the combined report
- `analysis_report`: compatibility alias of the combined report
- `memory_summary_report`: backward-compatible recent-memory summary
- `selector_context`: compact hints for search selection
- `meta_skill_context`: evidence and candidate skills for refinement/discovery/combination
