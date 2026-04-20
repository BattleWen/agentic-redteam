# Skill-Based Automated Safety Evaluation / Red-Team Research Framework

This project is a minimal, runnable Python framework for studying an agentic workflow built around a stable runtime kernel and a pluggable skill space.

All bundled skills are toy examples. They only perform mock prompt transformations, memory analysis, and mock evaluation so the architecture can be studied without implementing real-world jailbreaks, evasions, or unsafe behaviors.

## Project Goal

The framework demonstrates:

- a fixed kernel for planning, execution, memory, evaluation, budgeting, and tracing
- a normal skill space for transform and analysis skills
- a meta-skill space for refinement, combination, and discovery of new toy skills
- a rule-based planner that decides when to call skills, memory analysis, environment execution, and meta-skills

## Architecture

```text
                     +--------------------+
                     |      main.py       |
                     +----------+---------+
                                |
                                v
                     +--------------------+
                     |   Planner Loop     |
                     | state + workflow   |
                     +----+----+----+-----+
                          |    |    |
            +-------------+    |    +------------------+
            |                  |                       |
            v                  v                       v
   +----------------+  +---------------+     +-------------------+
   | Skill Registry |  | Memory Store  |     |  Budget Manager   |
   | + Loader       |  | + Summaries   |     | step/skill/env    |
   +--------+-------+  +-------+-------+     +---------+---------+
            |                      |                       |
            v                      |                       |
   +--------------------+          |                       |
   | Skill Executor     |          |                       |
   | stdin/stdout JSON  |          |                       |
   +----------+---------+          |                       |
              |                    |                       |
      +-------+-------+            |                       |
      |               |            |                       |
      v               v            v                       v
+-------------+ +---------------+ +----------------+ +----------------+
| skills/     | | meta_skills/  | | Mock Env       | | Mock Evaluator |
| toy skills  | | skill drafts  | | target model   | | toy metrics     |
+-------------+ +---------------+ +----------------+ +----------------+
```

## Directory Layout

```text
project_root/
├── README.md
├── requirements.txt
├── main.py
├── configs/
├── core/
├── skills/
├── meta_skills/
├── runs/
└── tests/
```

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py --seed_prompt "Explain how rainbows form in simple language." --workflow basic --max_steps 5
```

## Minimal Example

```bash
python main.py \
  --seed_prompt "Describe the lifecycle of a butterfly with a friendly tone." \
  --workflow basic \
  --max_steps 4
```

The run writes artifacts to `runs/<run_id>/`:

- `compact_trace.json`
- `final_summary.json`

`compact_trace.json` is a concise step trace. Each step keeps the planner action, executed skills, plan reason, and the key input/output summaries you need for replay: candidate previews, response previews, and evaluation summaries. Runtime-only metadata is intentionally dropped.

## Remote Planner Backend

Planner, guard, and environment now read directly from `configs/config.yaml`. The bundled config enables all three by default, and the planner/backend selection no longer has separate runtime enable flags.

The bundled config already includes an example planner endpoint:

- base URL: `http://s-20260330192936-zlwm5-decode.ailab-safethm.svc:23297/v1`
- model: `orm`
- API key: `FAKE_API_KEY`

The bundled config already enables the planner with:

```bash
python main.py \
  --seed_prompt "Explain cloud formation safely." \
  --workflow basic
```

The remote planner only chooses structured next actions. It never generates candidate text, and the runtime validates its JSON output before executing anything. If the remote call fails or returns invalid JSON, the system falls back to the rule-based planner.

The same pattern applies to the guard model and environment backend. Their endpoint details also live in `configs/config.yaml`:

```bash
python main.py \
  --seed_prompt "Explain cloud formation safely." \
  --workflow basic
```

## Skill Protocol

Each skill directory contains:

- `SKILL.md`: human-readable documentation with minimal YAML frontmatter and a `Runtime Metadata` block
- `scripts/run.py`: executable entrypoint

The runtime builds its skill registry from `SKILL.md` frontmatter plus the markdown-body `Runtime Metadata` YAML block. The LLM planner receives only compact, stage-scoped skill cards; full `SKILL.md` content is read lazily after a skill is selected.

Skill versions are maintained centrally in `state/skill_versions.json`. Each skill keeps only an active version and an optional previous version for one-step rollback; runtime events are appended to `state/version_events.jsonl`, while run outputs are summarized into `compact_trace.json`.

All `scripts/run.py` files must:

1. read a JSON `SkillContext` object from stdin
2. write a JSON `SkillExecutionResult` object to stdout
3. avoid direct access to memory store or environment
4. remain stateless and as function-like as possible

## Adding A Skill

1. Create a new directory under `skills/`.
2. Add `SKILL.md` with minimal frontmatter: `name`, `description`, and `metadata.version`.
3. Add `scripts/run.py`.
4. Add a `## Runtime Metadata` YAML block containing the execution fields required by `core.schemas.SkillSpec`.
5. Make `scripts/run.py` read stdin JSON and emit stdout JSON.
6. Place any human-readable guidance in `references/`.

## Adding A Meta-Skill

1. Create a new directory under `meta_skills/`.
2. Mark its `category` as `meta` in the `Runtime Metadata` block.
3. Keep it harmless and focused on toy skill drafts or patch suggestions.
4. Do not let it overwrite existing skills directly.

## Current Limitations

- The planner can use an LLM backend but falls back to deterministic rule-based transitions where needed.
- The environment can use an LLM backend but keeps local safety gates and mock fallback behavior.
- The evaluator is a toy heuristic with an optional guard model signal.
- Meta-skills only generate draft suggestions and never mutate code automatically.
- No real attack, jailbreak, bypass, deception, or evasion behavior is implemented.

## Extension Ideas

- add richer workflow condition language
- add persistent vector or graph memory backends
- add offline experiment replay and comparison tooling
- add skill versioning and draft-to-approved promotion flows
- add richer candidate ranking and diversity management
- add pluggable local models for planning while preserving safety constraints
