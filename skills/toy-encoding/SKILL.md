---
name: toy-encoding
description: Harmless formatting and encoding transformations for toy candidate generation.
metadata:
  version: "1.0"
---

# toy-encoding

## Runtime Metadata

```yaml
category: attack
stage: [search]
tags: [toy, encoding, transform]
inputs: [seed_prompt, constraints]
outputs: [candidates, artifacts]
entry: scripts/run.py
references: [references/patterns.md]
failure_modes:
  - encoded output loses readability
  - formatting adds little value
  - sanitized seed prompt becomes generic
family: toy-encoding
variant: encoding-core
status: active
applicability:
  prompt_buckets: [structured_request, general]
  target_traits: [chat, tool_like]
  memory_tags: [useful, needs_structure]
  preferred_stages: [search]
parameters_schema:
  type: object
  properties:
    encodings:
      type: array
      items:
        type: string
      default: [rot13_demo, base64_demo, json_wrap]
  required: []
retrieval_hints:
  lexical_triggers: [json, yaml, xml, schema, format, encode, base64, rot13]
  memory_keys: [risk_matrix, recent_risk_types, toy-encoding]
  prompt_buckets: [structured_request, general]
composition:
  compatible_families: [toy-persona, toy-paraphrase]
  conflicts_with: []
  pipeline_role: seed_transform
refinement:
  allowed_operations: [patch_suggestions, draft_variant]
  mutable_fields: [encoding_templates, wrappers, decoded_preview_style]
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

This harmless toy skill applies simple formatting or demo encodings to a sanitized seed prompt.

Allowed examples:

- JSON wrapping
- ROT13 demonstration
- Base64 demonstration

The output is intentionally non-actionable and only used to verify the framework execution path.
