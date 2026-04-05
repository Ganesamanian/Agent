# TODO: Add LLM Model Name as Langfuse Tags

## Steps:
- [x] Step 1: Edit agent_system/system.py - Added metadata={"llm_provider": ..., "llm_model": ...} to workflow.
- [x] Step 2: Edit agent_system/agents.py - Added metadata={"llm_provider": ..., "llm_model": ...} to all agent spans.
- [x] Step 3: Update this TODO.md with progress.
- [x] Step 4: Fixed to use 'metadata' (Langfuse SDK param) instead of 'tags'; now runs without error.
- [x] Complete: LLM provider and model name added as Langfuse metadata on all observations.
