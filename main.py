from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pathlib import Path

from agent_system import AppConfig, RunRequest, ThreeAgentSystem

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 3-agent terminal workflow.")
    parser.add_argument("--goal", default="", help="High-level complaint or support/legal question.")
    parser.add_argument("--case-context", default="", help="Optional extra context.")
    parser.add_argument("--provider", choices=["openai", "gemini"], default=None, help="LLM provider.")
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "gemini"],
        default=None,
        help="Embedding provider used by the already-built RAG collection.",
    )
    parser.add_argument("--public-url", action="append", default=[], help="Optional extra public URL.")
    parser.add_argument("--json", action="store_true", help="Print full JSON output instead of the pretty terminal view.")
    return parser.parse_args()


def status_symbol(status: str) -> str:
    return {"PENDING": "[ ]", "IN_PROGRESS": "[~]", "DONE": "[x]"}.get(status, "[?]")


def clear_screen() -> None:
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")


def render(goal: str, tasks: List[Dict[str, Any]], logs: List[str]) -> None:
    clear_screen()
    print(f"User input: {goal}\n")
    print("Agent")
    print("Todo")
    for task in tasks:
        print(f"{status_symbol(task.get('status', 'PENDING'))} {task.get('title', 'Task')}")
        notes = (task.get("notes") or "").strip()
        if notes and task.get("status") == "DONE":
            print(f"    -> {notes}")
    if not tasks:
        print("(planning...)")
    print("\nLog")
    for line in logs[-10:]:
        print(f"- {line}")
    if not logs:
        print("- waiting...")


def main() -> None:
    args = parse_args()
    goal = args.goal.strip() or input("Enter user goal: ").strip()
    if not goal:
        raise SystemExit("A goal is required.")

    config = AppConfig.from_env()
    system = ThreeAgentSystem(config)

    current_tasks: List[Dict[str, Any]] = []
    logs: List[str] = []
    session_id: Optional[str] = None

    def on_event(event: Dict[str, Any]) -> None:
        nonlocal current_tasks
        nonlocal session_id
        event_type = event.get("type")
        if event_type == "plan_created":
            current_tasks = event.get("plan", {}).get("tasks", [])
        elif event_type in {"task_started", "task_completed", "log"}:
            current_tasks = event.get("tasks", current_tasks)

        if event_type == "run_started":
            session_id = event.get('session_id')
            logs.append(
                f"run started | llm={event.get('llm_provider')} | embedding={event.get('embedding_provider')} | session={session_id}"
            )
        elif event_type == "log":
            logs.append(f"{event.get('agent', 'agent')}: {event.get('message', '')}")
        render(goal, current_tasks, logs)

    request = RunRequest(
        goal=goal,
        case_context=args.case_context,
        public_urls=args.public_url,
        user_id=config.default_user_id,
        llm_provider=args.provider,
        embedding_provider=args.embedding_provider,
    )

    try:
        result = system.run(request, event_handler=on_event)

        log_dir = Path("log") / session_id
        log_dir.mkdir(parents=True, exist_ok=True)

        (log_dir / "logs.json").write_text(json.dumps(logs, indent=2))
        (log_dir / "todos.json").write_text(json.dumps([task.to_dict() for task in result.tasks], indent=2))
        (log_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
        (log_dir / "result.txt").write_text(result.final_answer)

    except Exception as exc:
        print("\nRUN FAILED")
        print(str(exc))
        raise SystemExit(1)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print(result.final_answer)
    print(f"\nSession ID: {result.session_id}")
    print(f"LLM provider: {result.llm_provider} | Embedding provider: {result.embedding_provider}")


if __name__ == "__main__":
    main()

