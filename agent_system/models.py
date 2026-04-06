from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Task:
    id: str
    title: str
    owner: str
    status: str = "PENDING"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    evidence_id: str
    title: str
    excerpt: str
    source_type: str
    score: float = 0.0
    url: Optional[str] = None
    source_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentOutput:
    summary: str
    evidence: List[Evidence]
    cited_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "evidence": [item.to_dict() for item in self.evidence],
            "cited_sources": self.cited_sources,
        }


@dataclass
class RunRequest:
    goal: str
    case_context: str = ""
    public_urls: Optional[List[str]] = None
    user_id: str = "demo_user"
    llm_provider: Optional[str] = None
    embedding_provider: Optional[str] = None


@dataclass
class RunResult:
    session_id: str
    llm_provider: str
    embedding_provider: str
    tasks: List[Task]
    rag_output: AgentOutput
    web_output: AgentOutput
    action_plan: str
    next_steps: str
    final_answer: str
    cited_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "llm_provider": self.llm_provider,
            "embedding_provider": self.embedding_provider,
            "tasks": [task.to_dict() for task in self.tasks],
            "rag_output": self.rag_output.to_dict(),
            "web_output": self.web_output.to_dict(),
            "action_plan": self.action_plan,
            "next_steps": self.next_steps,
            "final_answer": self.final_answer,
            "cited_sources": self.cited_sources,
        }

