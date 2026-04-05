from __future__ import annotations

import uuid
from typing import Optional, Callable

from .config import AppConfig
from .models import RunRequest, RunResult, Task, AgentOutput
from .model_provider import ModelProvider
from .rag_store import RagStore
from .agents import RagRetrieverAgent, WebSearcherAgent, PlannerAgent
from .tracer import Tracer


class ThreeAgentSystem:
    def __init__(self, config: AppConfig):
        self.config = config
        self.provider = ModelProvider(config)
        self.tracer = Tracer(config.langfuse_public_key, config.langfuse_secret_key, config.langfuse_host, config.langfuse_timeout)
        self.rag_store = RagStore(config, self.provider)
        self.planner = PlannerAgent(config, self.provider, self.tracer)
        self.rag_agent = RagRetrieverAgent(config, self.provider, self.rag_store, self.tracer)
        self.web_agent = WebSearcherAgent(config, self.provider, self.tracer)

    def run(self, request: RunRequest, event_handler: Optional[Callable] = None) -> RunResult:
        def emit(payload):
            if event_handler:
                event_handler(payload)

        session_id = str(uuid.uuid4())
        llm_provider = (request.llm_provider or self.config.llm_provider or "openai").lower()
        embedding_provider = self.config.get_embedding_provider(request.embedding_provider)
        public_urls = request.public_urls or self.config.default_public_urls
        
        model_name = self.config.get_generation_model(llm_provider)
        
        trace_name = "agent_planner"

        emit({
            "type": "run_started",
            "session_id": session_id,
            "llm_provider": llm_provider,
            "embedding_provider": embedding_provider,
        })

        plan = self.planner.plan(request)
        tasks = [Task(**item) for item in plan["tasks"]]

        emit({
            "type": "plan_created",
            "plan": {"tasks": [task.to_dict() for task in tasks]},
        })

        with self.tracer.observation(
            name=trace_name,
            as_type="generation",
            input={
                "goal": request.goal,
                "case_context": request.case_context,
                "llm_provider": llm_provider,
                "embedding_provider": embedding_provider,
            },
            metadata={"llm_provider": llm_provider, "llm_model": model_name},
        ) as workflow:



            rag_output = AgentOutput(summary="", evidence=[])
            web_output = AgentOutput(summary="", evidence=[])
            action_plan = ""
            next_steps = ""
            final_answer = ""

            for task in tasks:
                task.status = "IN_PROGRESS"
                emit({"type": "task_started", "task": task.to_dict(), "tasks": [t.to_dict() for t in tasks]})

                if task.id == "rag":
                    emit({
                        "type": "log",
                        "agent": "agent-1",
                        "message": "sending task to Agent-2 (RAG retriever)",
                        "tasks": [t.to_dict() for t in tasks],
                    })
                    rag_output = self.rag_agent.run(
                        plan["rag_query"],
                        llm_provider=llm_provider,
                        embedding_provider=embedding_provider,
                    )
                    emit({
                        "type": "log",
                        "agent": "agent-2",
                        "message": f"retrieved {len(rag_output.evidence)} internal chunks",
                        "tasks": [t.to_dict() for t in tasks],
                    })

                elif task.id == "web":
                    emit({
                        "type": "log",
                        "agent": "agent-1",
                        "message": f"sending task to Agent-3 ({plan['tool_choice']})",
                        "tasks": [t.to_dict() for t in tasks],
                    })

                    web_output = self.web_agent.run(
                        request.goal,
                        keywords=plan["web_keywords"],
                        public_urls=public_urls,
                        tool=plan["tool_choice"],
                        llm_provider=llm_provider,
                    )
                    emit({
                        "type": "log",
                        "agent": "agent-3",
                        "message": f"collected {len(web_output.evidence)} public results using {plan['tool_choice']}",
                        "tasks": [t.to_dict() for t in tasks],
                    })

                elif task.id == "action":
                    emit({
                        "type": "log",
                        "agent": "agent-1",
                        "message": "combining public and internal evidence",
                        "tasks": [t.to_dict() for t in tasks],
                    })

                    action_plan = self.planner.write_action_plan(request, rag_output=rag_output, web_output=web_output)
                    emit({
                        "type": "log",
                        "agent": "agent-1",
                        "message": "action plan drafted",
                        "tasks": [t.to_dict() for t in tasks],
                    })

                elif task.id == "next_steps":
                    emit({
                        "type": "log",
                        "agent": "agent-1",
                        "message": "planning operational next steps",
                        "tasks": [t.to_dict() for t in tasks],
                    })

                    next_steps = self.planner.write_next_steps(
                        request,
                        rag_output=rag_output,
                        web_output=web_output,
                        action_plan=action_plan,
                    )
                    emit({
                        "type": "log",
                        "agent": "agent-1",
                        "message": "next steps drafted",
                        "tasks": [t.to_dict() for t in tasks],
                    })

                elif task.id == "result":
                    emit({
                        "type": "log",
                        "agent": "agent-1",
                        "message": "generating final result",
                        "tasks": [t.to_dict() for t in tasks],
                    })

                    final_answer = self.planner.write_final_answer(
                        request,
                        rag_output=rag_output,
                        web_output=web_output,
                        action_plan=action_plan,
                        next_steps=next_steps,
                    )
                    emit({
                        "type": "log",
                        "agent": "agent-1",
                        "message": "final result generated",
                        "tasks": [t.to_dict() for t in tasks],
                    })

                task.status = "DONE"
                emit({"type": "task_completed", "task": task.to_dict(), "tasks": [t.to_dict() for t in tasks]})

            if workflow:
                workflow.update(
                    output={
                        "final_answer": final_answer,
                        "rag_chunks": len(rag_output.evidence),
                        "web_results": len(web_output.evidence),
                    }
                )

            self.tracer.update_current_trace(
                output={
                    "final_answer": final_answer,
                    "rag_chunks": len(rag_output.evidence),
                    "web_results": len(web_output.evidence),
                }
            )

        self.tracer.flush()

        return RunResult(
            session_id=session_id,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            tasks=tasks,
            rag_output=rag_output,
            web_output=web_output,
            action_plan=action_plan,
            next_steps=next_steps,
            final_answer=final_answer,
        )

