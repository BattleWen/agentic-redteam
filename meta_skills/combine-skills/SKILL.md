---
name: combine-skills
description: Draft a new harmless toy skill by combining two existing toy skills.
metadata:
  version: "1.0"
---

# combine-skills

## Runtime Metadata

```yaml
category: meta
stage: [escalation, refine]
tags: [toy, meta, combine]
inputs: [extra.target_skill_specs, evaluator_feedback]
outputs: [artifacts]
entry: scripts/run.py
references: [references/rules.md]
failure_modes:
  - combined skill is redundant
  - draft is too vague
  - input skill specs are incomplete
family: combine-skills
variant: combine-core
status: active
applicability:
  prompt_buckets: [general]
  target_traits: []
  memory_tags: [needs_combination, high_refusal, useful]
  preferred_stages: [refine, escalation]
parameters_schema:
  type: object
  properties:
    skill_names:
      type: array
      items:
        type: string
      minItems: 2
  required: []
retrieval_hints:
  lexical_triggers: [combine, hybrid, pair, compose]
  memory_keys: [target_skill_specs, recent_memory, skill_counts]
  prompt_buckets: [general]
composition:
  compatible_families: [toy-persona, toy-paraphrase, toy-encoding]
  conflicts_with: []
  pipeline_role: meta_composer
refinement:
  allowed_operations: [draft_variant, composition_patch]
  mutable_fields: [composition, candidate_logic, description]
  promotion_metric: avg_overall_score
  rollback_metric: avg_overall_score
evaluation_focus: [diversity_score, usefulness_score]
safety_scope:
  mode: harmless_mock_only
  disallowed_content:
    - real_jailbreak_instructions
    - real_bypass_workflows
    - malware_or_weapon_content
```

This harmless meta-skill combines two toy skills into a new draft concept.

Example:

- persona framing + paraphrase style
- encoding wrapper + explanatory tone

The output is a draft spec idea, not an executable new skill directory.
