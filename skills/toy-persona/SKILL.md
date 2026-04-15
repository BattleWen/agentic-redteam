---
name: toy-persona
description: Harmless persona-style prompt rewriting for toy candidate generation.
metadata:
  version: "1.0"
---

# toy-persona

## Runtime Metadata

```yaml
category: attack
stage: [search]
tags: [toy, persona, transform]
inputs: [seed_prompt, constraints]
outputs: [candidates, artifacts]
entry: scripts/run.py
references: [references/patterns.md, references/examples.md]
failure_modes:
  - low diversity across personas
  - overly generic outputs
  - filtered seed prompt reduces specificity
family: toy-persona
variant: persona-core
status: active
applicability:
  prompt_buckets: [roleplay_style, educational_explanation, safety_sensitive, general]
  target_traits: [chat, instruction_following]
  memory_tags: [useful, needs_diversity]
  preferred_stages: [search]
parameters_schema:
  type: object
  properties:
    persona_count:
      type: integer
      minimum: 1
      maximum: 5
      default: 3
  required: []
retrieval_hints:
  lexical_triggers: [role, persona, teacher, friendly, guide, librarian, curator]
  memory_keys: [skill_counts, recent_skill_names, toy-persona]
  prompt_buckets: [roleplay_style, educational_explanation]
composition:
  compatible_families: [toy-paraphrase, toy-encoding]
  conflicts_with: []
  pipeline_role: seed_transform
refinement:
  allowed_operations: [patch_suggestions, draft_variant]
  mutable_fields: [templates, persona_count, rationale_style]
  promotion_metric: avg_overall_score
  rollback_metric: avg_overall_score
evaluation_focus: [usefulness_score, diversity_score]
safety_scope:
  mode: harmless_mock_only
  disallowed_content:
    - real_jailbreak_instructions
    - real_bypass_workflows
    - malware_or_weapon_content
```

This harmless toy skill rewrites the seed prompt into a few role-framed variants.

It is only a structural example for the framework:

- no real bypass behavior
- no unsafe instructions
- blocked terms are replaced with a neutral placeholder topic

## Input

- `SkillContext.seed_prompt`

## Output

- `SkillExecutionResult.candidates`

## Notes

The generated personas are intentionally benign, such as teacher, curator, or librarian voices.
