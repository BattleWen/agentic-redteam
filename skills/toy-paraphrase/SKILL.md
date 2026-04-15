---
name: toy-paraphrase
description: Harmless paraphrase generator for toy candidate search.
metadata:
  version: "1.0"
---

# toy-paraphrase

## Runtime Metadata

```yaml
category: attack
stage: [search]
tags: [toy, paraphrase, transform]
inputs: [seed_prompt, constraints]
outputs: [candidates, artifacts]
entry: scripts/run.py
references: [references/patterns.md]
failure_modes:
  - paraphrases are too similar
  - paraphrases become overly abstract
  - sanitized topic reduces nuance
family: toy-paraphrase
variant: paraphrase-core
status: active
applicability:
  prompt_buckets: [rewrite_request, educational_explanation, safety_sensitive, general]
  target_traits: [chat, instruction_following]
  memory_tags: [useful, needs_clarity]
  preferred_stages: [search]
parameters_schema:
  type: object
  properties:
    paraphrase_count:
      type: integer
      minimum: 1
      maximum: 5
      default: 3
  required: []
retrieval_hints:
  lexical_triggers: [rewrite, rephrase, paraphrase, restate, plain-language, summary]
  memory_keys: [recent_skill_names, risk_matrix, toy-paraphrase]
  prompt_buckets: [rewrite_request, educational_explanation]
composition:
  compatible_families: [toy-persona, toy-encoding]
  conflicts_with: []
  pipeline_role: seed_transform
refinement:
  allowed_operations: [patch_suggestions, draft_variant]
  mutable_fields: [templates, paraphrase_count, tone_labels]
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

This harmless toy skill produces simple paraphrases of a sanitized seed prompt.

It aims to provide:

- varied wording
- stable meaning at a high level
- no unsafe content
