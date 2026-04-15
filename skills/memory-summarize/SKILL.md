---
name: memory-summarize
description: Summarize recent memory entries for harmless workflow escalation.
metadata:
  version: "1.0"
---

# memory-summarize

## Runtime Metadata

```yaml
category: analysis
stage: [analysis, escalation]
tags: [toy, memory, summary]
inputs: [memory_summary, extra.recent_memory]
outputs: [artifacts]
entry: scripts/run.py
references: [references/notes.md]
failure_modes:
  - summary too shallow
  - small memory windows hide patterns
  - counts omit qualitative nuance
family: memory-summarize
variant: summary-core
status: active
applicability:
  prompt_buckets: [general]
  target_traits: []
  memory_tags: [high_refusal, failure_pattern, useful]
  preferred_stages: [analysis, escalation]
parameters_schema:
  type: object
  properties:
    window_size:
      type: integer
      minimum: 1
      default: 5
  required: []
retrieval_hints:
  lexical_triggers: [memory, summary, recent]
  memory_keys: [recent_memory, risk_matrix, skill_counts, recent_risk_types]
  prompt_buckets: [general]
composition:
  compatible_families: [retrieval-analysis]
  conflicts_with: []
  pipeline_role: memory_summary
refinement:
  allowed_operations: [patch_suggestions, draft_variant]
  mutable_fields: [summary_fields, aggregation_rules]
  promotion_metric: avg_overall_score
  rollback_metric: avg_overall_score
evaluation_focus: [summary_quality]
safety_scope:
  mode: harmless_mock_only
  disallowed_content:
    - real_jailbreak_instructions
    - real_bypass_workflows
    - malware_or_weapon_content
```

This harmless analysis skill summarizes recent memory entries passed in through `SkillContext.extra.recent_memory`.

It does not read the memory store directly.
