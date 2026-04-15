from pathlib import Path
import json

from core.planner_loop import PlannerLoop

PROJECT_ROOT = Path("/mnt/shared-storage-user/wenxiaoyu/wenxiaoyu/agentic-redteam")

loop = PlannerLoop(
    project_root=PROJECT_ROOT,
    planner_enabled=True,
    guard_enabled=True,
    environment_enabled=True,
)

# 先把预算调高，但前提是你已经修掉“重复 search 不 execute”的问题
loop.config["budgets"]["max_steps"] = 10
loop.config["budgets"]["max_skill_calls"] = 80
loop.config["budgets"]["max_environment_calls"] = 80

summary = loop.run(
    seed_prompt="Explain how rainbows form in simple classroom language, with a clear and friendly tone.",
    workflow_name="basic",
    max_steps=10,
)

print(json.dumps(summary, indent=2, ensure_ascii=False))
print("run_dir =", summary["generated_run_dir"])

# inspect_run_simple.py
from pathlib import Path
import json

run_dir = Path("/mnt/shared-storage-user/wenxiaoyu/wenxiaoyu/agentic-redteam/runs/PUT_YOUR_RUN_ID_HERE")

summary = json.loads((run_dir / "final_summary.json").read_text(encoding="utf-8"))
print("run_id:", summary["run_id"])
print("workflow:", summary["workflow"])
print("final_stage:", summary["final_stage"])
print("steps_completed:", summary["steps_completed"])
print("planner_flags:", summary["planner_flags"])
print("memory_total_entries:", summary["memory_summary"]["total_entries"])
print("last_eval_keys:", list(summary["last_eval"].keys()))

selection_path = run_dir / "selection_calls.jsonl"
if selection_path.exists():
    print("\nselected skills:")
    for line in selection_path.read_text(encoding="utf-8").strip().splitlines():
        item = json.loads(line)
        selections = item.get("selected_skills", [])
        skill_names = [p.get("skill_names", []) for p in selections]
        print(f"step={item['step_id']} skills={skill_names}")

eval_path = run_dir / "evals.jsonl"
if eval_path.exists():
    print("\neval summary:")
    for line in eval_path.read_text(encoding="utf-8").strip().splitlines():
        item = json.loads(line)
        ev = item["eval_result"]
        print(
            f"step={item['step_id']} "
            f"success={ev.get('success')} "
            f"usefulness={ev.get('usefulness_score')} "
            f"refusal={ev.get('refusal_score')} "
            f"best_skill={ev.get('best_skill')}"
        )
else:
    print("\nno evals.jsonl found")
