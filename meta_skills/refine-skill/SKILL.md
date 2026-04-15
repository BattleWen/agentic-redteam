---
name: refine-skill
description: Create harmless patch suggestions or draft variants for toy skills.
metadata:
  version: "1.0"
---

# refine-skill

## Runtime Metadata

```yaml
category: meta
stage: [refine, escalation]
tags: [toy, meta, refine]
inputs: [extra.target_skill_spec, evaluator_feedback, extra.recent_memory]
outputs: [artifacts]
entry: scripts/run.py
references: [references/rules.md]
failure_modes:
  - suggestions are too generic
  - draft is not meaningfully different
  - feedback may be too sparse
family: refine-skill
variant: refine-core
status: active
applicability:
  prompt_buckets: [general]
  target_traits: []
  memory_tags: [high_refusal, useful, needs_refinement]
  preferred_stages: [refine, escalation]
parameters_schema:
  type: object
  properties:
    target_family:
      type: string
  required: []
retrieval_hints:
  lexical_triggers: [refine, improve, patch, polish]
  memory_keys: [evaluator_feedback, target_skill_spec, recent_memory]
  prompt_buckets: [general]
composition:
  compatible_families: [toy-persona, toy-paraphrase, toy-encoding]
  conflicts_with: []
  pipeline_role: meta_refiner
refinement:
  allowed_operations: [patch_suggestions, draft_variant, promotion_recommendation]
  mutable_fields: [description, templates, candidate_logic, rationale_style]
  promotion_metric: avg_overall_score
  rollback_metric: avg_overall_score
evaluation_focus: [usefulness_score, refusal_score]
safety_scope:
  mode: harmless_mock_only
  disallowed_content:
    - real_jailbreak_instructions
    - real_bypass_workflows
    - malware_or_weapon_content
```

This harmless meta-skill proposes a draft refinement for an existing toy skill.

It does not modify the original skill on disk.

Typical output:

- patch suggestion bullets
- a new draft skill name
- a short rationale tied to recent evaluator feedback
