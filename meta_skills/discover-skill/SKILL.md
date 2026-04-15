---
name: discover-skill
description: Draft a new harmless toy skill concept from repeated failure patterns.
metadata:
  version: "1.0"
---

# discover-skill

## Runtime Metadata

```yaml
category: meta
stage: [escalation]
tags: [toy, meta, discover]
inputs: [memory_summary, extra.recent_memory, evaluator_feedback]
outputs: [artifacts]
entry: scripts/run.py
references: [references/rules.md]
failure_modes:
  - draft is too speculative
  - repeated failures may be coincidental
  - proposed logic may overlap existing toy skills
family: discover-skill
variant: discover-core
status: active
applicability:
  prompt_buckets: [general]
  target_traits: []
  memory_tags: [repeated_failure, high_refusal, analysis_needed]
  preferred_stages: [escalation]
parameters_schema:
  type: object
  properties:
    failure_window:
      type: integer
      minimum: 1
      default: 5
  required: []
retrieval_hints:
  lexical_triggers: [discover, new skill, failure pattern, draft]
  memory_keys: [recent_failure_tags, recent_memory, memory_summary]
  prompt_buckets: [general]
composition:
  compatible_families: [retrieval-analysis, memory-summarize, refine-skill]
  conflicts_with: []
  pipeline_role: meta_discovery
refinement:
  allowed_operations: [draft_variant, new_family_proposal]
  mutable_fields: [description, candidate_logic, triggering_patterns]
  promotion_metric: avg_overall_score
  rollback_metric: avg_overall_score
evaluation_focus: [diversity_score, refusal_score]
safety_scope:
  mode: harmless_mock_only
  disallowed_content:
    - real_jailbreak_instructions
    - real_bypass_workflows
    - malware_or_weapon_content
```

This harmless meta-skill turns repeated toy failure patterns into a draft for a new toy skill.

It is a structural example of skill discovery, not a real attack generator.
