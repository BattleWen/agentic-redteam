---
name: retrieval-analysis
description: Analyze recent toy failures and memory patterns.
metadata:
  version: "1.0"
---

# retrieval-analysis

## Runtime Metadata

```yaml
category: analysis
stage: [analysis, escalation]
tags: [toy, retrieval, analysis]
inputs: [memory_summary, evaluator_feedback, extra.recent_memory]
outputs: [artifacts]
entry: scripts/run.py
references: [references/notes.md]
failure_modes:
  - pattern analysis overfits small samples
  - few entries reduce confidence
  - feedback can be noisy
family: retrieval-analysis
variant: analysis-core
status: active
applicability:
  prompt_buckets: [general]
  target_traits: []
  memory_tags: [high_refusal, failure_pattern, analysis_needed]
  preferred_stages: [analysis, escalation]
parameters_schema:
  type: object
  properties:
    top_k_patterns:
      type: integer
      minimum: 1
      default: 3
  required: []
retrieval_hints:
  lexical_triggers: [pattern, analysis, failure, retrieval]
  memory_keys: [recent_memory, risk_matrix, recent_risk_types, evaluator_feedback]
  prompt_buckets: [general]
composition:
  compatible_families: [memory-summarize, refine-skill, discover-skill]
  conflicts_with: []
  pipeline_role: memory_analysis
refinement:
  allowed_operations: [patch_suggestions, draft_variant]
  mutable_fields: [observation_rules, tag_grouping]
  promotion_metric: avg_overall_score
  rollback_metric: avg_overall_score
evaluation_focus: [refusal_score, usefulness_score]
safety_scope:
  mode: harmless_mock_only
  disallowed_content:
    - real_jailbreak_instructions
    - real_bypass_workflows
    - malware_or_weapon_content
```

This harmless analysis skill looks at recent memory and evaluator feedback and extracts toy patterns such as:

- repeated refusal tags
- which toy skills were recently used
- whether usefulness looks flat or improving
