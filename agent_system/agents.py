from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from typing import Tuple, List, Optional, Dict, Any

from .config import AppConfig
from .model_provider import ModelProvider
from .models import AgentOutput, Evidence, RunRequest
from .rag_store import RagStore
from .tracer import Tracer
from .utils import shorten, sanitize_text, split_sentences, extract_keywords, lexical_overlap_score

class RagRetrieverAgent:
    def __init__(self, config: AppConfig, provider: ModelProvider, rag_store: RagStore, tracer: Tracer):
        self.config = config
        self.provider = provider
        self.rag_store = rag_store
        self.tracer = tracer

    def run(self, query: str, *, llm_provider: Optional[str], embedding_provider: Optional[str]) -> AgentOutput:
        model_name = self.config.get_generation_model(llm_provider)
        with self.tracer.observation(
            name="agent_rag_retriever",
            as_type="retriever",
            input={"query": query, "embedding_provider": embedding_provider},
            metadata={"llm_provider": llm_provider, "llm_model": model_name},
        ) as obs:
            evidence = self.rag_store.retrieve(
                query,
                top_k=self.config.rag_top_k,
                embedding_provider=embedding_provider,
            )
            summary = self._summarize(query, evidence, llm_provider=llm_provider)
            cited_sources = list(set(e.source_path or e.url or e.title for e in evidence if e.source_path or e.url))
            output = AgentOutput(summary=summary, evidence=evidence, cited_sources=cited_sources)
            if obs:
                obs.update(output={
                    "summary": summary,
                    "evidence_count": len(evidence),
                    "sources": [e.source_path for e in evidence],
                })
            return output

    def _summarize(self, query: str, evidence: List[Evidence], *, llm_provider: Optional[str]) -> str:
        if not evidence:
            return "No internal policy chunks were retrieved."
        context = "\n".join(f"- {item.title}: {item.excerpt}" for item in evidence[:3])
        # system_prompt = ( "You are Agent-2, the RAG retriever agent. Summarize only the retrieved internal policy evidence. " 
        #                  "Do not invent rules. Mention uncertainty when evidence is incomplete." ) 
        # user_prompt = f"Task: {query}\n\nRetrieved internal evidence:\n{context}\n\nWrite 3 concise bullets."
        system_prompt, user_prompt = self.tracer.get_prompt_from_langfuse("RagAgentPrompt", query, context)
        citation_instruction = "\n\nIMPORTANT: Cite sources inline as [1], [2]. End with 'Sources: [1] {title1}, [2] {title2}' using top 3."
        user_prompt += citation_instruction.format(title1=evidence[0].title if evidence else '', title2=evidence[1].title if len(evidence)>1 else '')
        text = self.provider.generate_text(provider=llm_provider, system_prompt=system_prompt, user_prompt=user_prompt)
        if text.strip():
            return text.strip()
        bullets = [f"- {item.title}: {item.excerpt}" for item in evidence[:3]]
        return "\n".join(bullets)


class WebSearcherAgent:
    def __init__(self, config: AppConfig, provider: ModelProvider, tracer: Tracer):
        self.config = config
        self.provider = provider
        self.tracer = tracer

    def run(self, query: str, *, keywords, public_urls, tool, llm_provider):
        model_name = self.config.get_generation_model(llm_provider)
        with self.tracer.observation(
            name="agent_websearcher",
            as_type="tool",
            input={"query": query, "keywords": keywords, "tool": tool},
            metadata={"llm_provider": llm_provider, "llm_model": model_name},
        ) as obs:
            if tool == "serpapi" and self.config.serpapi_api_key:
                evidence = self._search_with_serpapi(query, keywords)
                if not evidence:
                    evidence = self._search_with_beautifulsoup(keywords, public_urls)
            else:
                evidence = self._search_with_beautifulsoup(keywords, public_urls)

            summary = self._summarize(query, evidence, llm_provider=llm_provider)
            cited_sources = list(set(e.source_path or e.url or e.title for e in evidence if e.source_path or e.url))
            output = AgentOutput(summary=summary, evidence=evidence, cited_sources=cited_sources)

            if obs:
                obs.update(output={
                    "summary": summary,
                    "evidence_count": len(evidence),
                    "urls": [e.url for e in evidence if e.url],
                })
            return output

    def _search_with_serpapi(self, query: str, keywords: List[str]) -> List[Evidence]:
        params = {
            "engine": "google",
            "q": f"site:booking.com {query} {' '.join(keywords[:4])}",
            "api_key": self.config.serpapi_api_key,
            "num": self.config.web_top_k,
        }
        try:
            response = requests.get("https://serpapi.com/search.json", params=params, timeout=self.config.request_timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []

        evidence: List[Evidence] = []
        for idx, item in enumerate(payload.get("organic_results", [])[: self.config.web_top_k], start=1):
            link = item.get("link") or item.get("redirect_link")
            title = item.get("title") or f"Search result {idx}"
            snippet = item.get("snippet") or ""
            evidence.append(
                Evidence(
                    evidence_id=f"web_{idx}",
                    title=title,
                    excerpt=shorten(snippet, 280),
                    source_type="web",
                    url=link,
                    score=float(self.config.web_top_k - idx + 1),
                )
            )
        return evidence

    def _fetch_page_text(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=self.config.request_timeout, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
        except Exception:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text(" ", strip=True)
        from .utils import sanitize_text
        return sanitize_text(text)

    def _search_with_beautifulsoup(self, keywords: List[str], public_urls: List[str]) -> List[Evidence]:
        evidence: List[Evidence] = []
        for url in public_urls[: max(1, self.config.web_top_k)]:
            page_text = self._fetch_page_text(url)
            if not page_text:
                continue
            sentences = split_sentences(page_text)
            scored: List[Tuple[float, str]] = []
            for sentence in sentences:
                score = lexical_overlap_score(sentence, keywords)
                if score > 0:
                    scored.append((score, sentence))
            scored.sort(key=lambda item: (-item[0], len(item[1])))
            if scored:
                best_excerpt = " ".join(text for _, text in scored[:2])
                evidence.append(
                    Evidence(
                        evidence_id=f"web_{len(evidence)+1}",
                        title=url,
                        excerpt=shorten(best_excerpt, 320),
                        source_type="web",
                        url=url,
                        score=float(scored[0][0]),
                    )
                )
        return evidence[: self.config.web_top_k]

    def _summarize(self, query: str, evidence: List[Evidence], *, llm_provider: Optional[str]) -> str:
        if not evidence:
            return "No public Booking.com evidence was found from the configured web sources."
        context = "\n".join(f"- {item.title}: {item.excerpt}" for item in evidence[:3])
        # system_prompt = (
        #     "You are Agent-3, the websearcher. Summarize only the public Booking.com evidence. "
        #     "Do not mention internal policy. Keep it short and factual."
        # )
        # user_prompt = f"Task: {query}\n\nPublic evidence:\n{context}\n\nWrite 3 concise bullets."
        system_prompt, user_prompt = self.tracer.get_prompt_from_langfuse("WebAgentPrompt", query, context)
        citation_instruction = "\n\nIMPORTANT: Cite sources inline as [1], [2]. End with 'Sources: [1] {url1}, [2] {url2}' using top 3."
        user_prompt += citation_instruction.format(url1=evidence[0].url if evidence else '', url2=evidence[1].url if len(evidence)>1 else '')
        text = self.provider.generate_text(provider=llm_provider, system_prompt=system_prompt, user_prompt=user_prompt)
        if text.strip():
            return text.strip()
        bullets = [f"- {item.title}: {item.excerpt}" for item in evidence[:3]]
        return "\n".join(bullets)


class PlannerAgent:
    TASK_TEMPLATES = [
        {"id": "rag", "title": "Check policy in RAG", "owner": "agent-2"},
        {"id": "web", "title": "Check website", "owner": "agent-3"},
        {"id": "action", "title": "Plan the action", "owner": "agent-1"},
        {"id": "next_steps", "title": "Plan next steps", "owner": "agent-1"},
        {"id": "result", "title": "Generate result", "owner": "agent-1"},
    ]

    def __init__(self, config: AppConfig, provider: ModelProvider, tracer):
        self.config = config
        self.provider = provider
        self.tracer = tracer

    def _default_plan(self, request: RunRequest) -> Dict[str, Any]:
        return {
            "goal_summary": sanitize_text(request.goal),
            "rag_query": sanitize_text(f"{request.goal}. {request.case_context}"),
            "web_keywords": extract_keywords(f"{request.goal} {request.case_context}"),
            "tool_choice": "beautifulsoup",
            "tasks": [dict(item) for item in self.TASK_TEMPLATES],
        }

    def _normalize_plan(self, candidate: Optional[Dict[str, Any]], fallback: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            "goal_summary": sanitize_text((candidate or {}).get("goal_summary") or fallback["goal_summary"]),
            "rag_query": sanitize_text((candidate or {}).get("rag_query") or fallback["rag_query"]),
            "web_keywords": list(fallback["web_keywords"]),
            "tool_choice": fallback["tool_choice"],
            "tasks": [dict(item) for item in self.TASK_TEMPLATES],
        }

        raw_keywords = (candidate or {}).get("web_keywords")
        if isinstance(raw_keywords, list):
            cleaned_keywords = [sanitize_text(str(item)) for item in raw_keywords if sanitize_text(str(item))]
            if cleaned_keywords:
                normalized["web_keywords"] = cleaned_keywords[:6]

        raw_tool = str((candidate or {}).get("tool_choice") or "").strip().lower()
        if raw_tool in {"serpapi", "beautifulsoup"}:
            normalized["tool_choice"] = raw_tool

        raw_tasks = (candidate or {}).get("tasks")
        if isinstance(raw_tasks, list) and len(raw_tasks) == len(self.TASK_TEMPLATES):
            rebuilt_tasks: List[Dict[str, Any]] = []
            for template, raw_task in zip(self.TASK_TEMPLATES, raw_tasks):
                if not isinstance(raw_task, dict):
                    rebuilt_tasks = []
                    break
                rebuilt_tasks.append(
                    {
                        "id": template["id"],
                        "title": sanitize_text(str(raw_task.get("title") or raw_task.get("task") or raw_task.get("name") or template["title"])),
                        "owner": sanitize_text(str(raw_task.get("owner") or template["owner"])),
                    }
                )
            if rebuilt_tasks:
                normalized["tasks"] = rebuilt_tasks

        return normalized

    def plan(self, request: RunRequest) -> Dict[str, Any]:
        fallback = self._default_plan(request)

        system_prompt = (
            "You are Agent-1, the planner and orchestrator. "
            "Return only valid JSON with keys: goal_summary, rag_query, web_keywords, tool_choice, tasks. "
            "Use exactly these 5 task ids in this order: rag, web, action, next_steps, result. "
            "Each task must include id, title, and owner. "
            "tool_choice must be either 'serpapi' or 'beautifulsoup'."
        )

        user_prompt = (
            f"Goal: {request.goal}\n"
            f"Case context: {request.case_context or '(none)'}\n\n"
            "Create a compact plan for a Booking.com complaint triage run."
        )

        model_name = self.config.get_generation_model(request.llm_provider)
        with self.tracer.observation(
            name="agent_planner_plan",
            as_type="span",
            input={"goal": request.goal, "case_context": request.case_context},
            metadata={"llm_provider": request.llm_provider, "llm_model": model_name},
        ) as obs:
            planned = self.provider.generate_json(
                provider=request.llm_provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            final_plan = self._normalize_plan(planned, fallback)
            if obs:
                obs.update(output=final_plan)
            return final_plan

    def write_action_plan(
        self,
        request: RunRequest,
        *,
        rag_output: AgentOutput,
        web_output: AgentOutput,
    ) -> str:
        model_name = self.config.get_generation_model(request.llm_provider)
        with self.tracer.observation(
            name="agent_planner_action",
            as_type="span",
            input={"goal": request.goal},
            metadata={"llm_provider": request.llm_provider, "llm_model": model_name},
        ) as obs:
            system_prompt = (
                "You are Agent-1. Combine internal and public evidence into a recommended action plan. "
                "Be precise, grounded, and mention uncertainty if evidence conflicts."
            )
            user_prompt = (
                f"Goal: {request.goal}\n\n"
                f"Internal evidence summary:\n{rag_output.summary}\n\n"
                f"Public evidence summary:\n{web_output.summary}\n\n"
                "Write a concise action plan in 3 bullets."
            )
            text = self.provider.generate_text(
                provider=request.llm_provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            
            if obs:
                obs.update(output={"action_plan": text})
            return text.strip()

    def write_next_steps(
        self,
        request: RunRequest,
        *,
        rag_output: AgentOutput,
        web_output: AgentOutput,
        action_plan: str,
    ) -> str:
        model_name = self.config.get_generation_model(request.llm_provider)
        with self.tracer.observation(
            name="agent_planner_next_steps",
            as_type="span",
            input={"goal": request.goal},
            metadata={"llm_provider": request.llm_provider, "llm_model": model_name},
        ) as obs:
            system_prompt = (
                "You are Agent-1. Write operational next steps for a support/legal triage workflow. "
                "Focus on missing facts, escalation, and what the operator should do next."
            )
            user_prompt = (
                f"Goal: {request.goal}\n\n"
                f"Action plan:\n{action_plan}\n\n"
                f"Internal evidence:\n{rag_output.summary}\n\n"
                f"Public evidence:\n{web_output.summary}\n\n"
                "Write 4 short next steps."
            )
            text = self.provider.generate_text(
                provider=request.llm_provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            
            if obs:
                obs.update(output={"next_steps": text})
            return text.strip()

    def write_final_answer(
        self,
        request: RunRequest,
        *,
        rag_output: AgentOutput,
        web_output: AgentOutput,
        action_plan: str,
        next_steps: str,
    ) -> str:
        model_name = self.config.get_generation_model(request.llm_provider)
        with self.tracer.observation(
            name="agent_planner_final",
            as_type="span",
            input={"goal": request.goal},
            metadata={"llm_provider": request.llm_provider, "llm_model": model_name},
        ) as obs:
            system_prompt = (
                "You are Agent-1. Produce a final user-facing triage note. "
                "Structure it as Issue, Internal policy, Public policy, Recommended action, Next steps."
            )
            user_prompt = (
                f"Goal: {request.goal}\n\n"
                f"Internal policy summary:\n{rag_output.summary}\n\n"
                f"Public policy summary:\n{web_output.summary}\n\n"
                f"Action plan:\n{action_plan}\n\n"
                f"Next steps:\n{next_steps}"
            )
            text = self.provider.generate_text(
                provider=request.llm_provider,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            
            if obs:
                obs.update(output={"final_answer": text})
            return text.strip()

