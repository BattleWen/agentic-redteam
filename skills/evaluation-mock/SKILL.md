---
name: evaluation-mock
description: Composite toy evaluation summary skill with optional safety guard annotations.
metadata:
  version: "2.0"
---

# evaluation-mock

## Runtime Metadata

```yaml
category: evaluation
stage: [evaluation]
tags: [toy, evaluation, analysis]
inputs: [prior_candidates, extra.last_responses, extra.precomputed_eval, seed_prompt]
outputs: [artifacts]
entry: scripts/run.py
references: [references/metrics.md]
failure_modes:
  - guard model may be unavailable
  - ranking is still heuristic at small scale
  - notes can duplicate core evaluator output
family: evaluation-mock
variant: evaluation-core
status: active
applicability:
  prompt_buckets: [general, structured_request, rewrite_request, roleplay_style, educational_explanation, safety_sensitive]
  target_traits: [chat, instruction_following, guarded]
  memory_tags: [evaluation_ready, high_refusal, useful]
  preferred_stages: [evaluation]
parameters_schema:
  type: object
  properties:
    emit_ranked_summary:
      type: boolean
      default: true
  required: []
retrieval_hints:
  lexical_triggers: [evaluate, score, guard, risk, rank]
  memory_keys: [score_bundles, guard_backend, best_candidate_index]
  prompt_buckets: [general, structured_request, rewrite_request]
composition:
  compatible_families: [toy-persona, toy-paraphrase, toy-encoding, memory-summarize, retrieval-analysis]
  conflicts_with: []
  pipeline_role: evaluation_summary
refinement:
  allowed_operations: [patch_suggestions, draft_variant]
  mutable_fields: [ranking_notes, summary_fields, guard_annotations]
  promotion_metric: asr
  rollback_metric: asr
evaluation_focus: [refusal_score, request_risk_score, response_risk_score]
safety_scope:
  mode: harmless_mock_only
  disallowed_content:
    - real_jailbreak_instructions
    - real_bypass_workflows
    - malware_or_weapon_content
```

This harmless evaluation skill supplements the kernel evaluator by generating human-readable notes about:

- which toy strategies were used
- whether the mock response styles looked diverse
- which candidate appears best in a toy sense
- whether an optional safety guard model flagged additional risk

It does not replace the kernel evaluator. It summarizes the evaluator's score bundle and any available guard annotations.
