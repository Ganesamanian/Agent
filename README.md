# Three-Agent CLI (Planner + RAG + Web)

A minimal terminal-first agent workflow for complaint triage, combining planning, RAG retrieval, and web evidence collection.

## Overview

![Agent Flow](image/Agent_flow.png)

This repository implements a simple three-agent system for Booking.com-like customer issues.

- **Agent-1 Planner** creates a structured plan from a user goal.
- **Agent-2 RAG Retriever** fetches internal policy evidence from a Milvus Lite vector store.
- **Agent-3 Websearcher** gathers public evidence from Booking.com pages using SerpAPI or BeautifulSoup.

The system is designed for live CLI execution with real-time plan/log updates and a separate RAG build/evaluation flow.

## What it does

- Generates a TODO-style plan from a customer goal.
- Executes tasks sequentially through a planner-driven loop.
- Uses real tools: Milvus Lite vector retrieval and web scraping / search.
- Summarizes retrieved evidence and generates a final user-facing triage response.

## Key Capabilities

- **Planning:** `PlannerAgent.plan()` generates a structured JSON plan with:
  - `rag_query`
  - `web_keywords`
  - `tool_choice`
  - `tasks` list with fixed IDs `rag`, `web`, `action`, `next_steps`, `result`
- **Execution loop:** `ThreeAgentSystem.run()` iterates tasks sequentially, updating task status and emitting events.
- **Tool use:** External tool use includes:
  - Milvus Lite / vector retrieval for internal RAG evidence
  - BeautifulSoup scraping of public URLs
  - Optional SerpAPI search when configured
- **Context strategy:** The system avoids prompt overflow by:
  - extracting concise `web_keywords` instead of sending full text
  - retrieving only top-k evidence chunks from RAG
  - summarizing evidence before using it in later planner prompts
  - building the vector store separately with `run_milvius.py`

## Project Structure

```
.
├── agent_system/          # Core agents and utilities
│   ├── __init__.py
│   ├── agents.py          # Agent-1/2/3 implementations
│   ├── config.py          # AppConfig (env, providers, settings)
│   ├── model_provider.py  # OpenAI/Gemini abstraction layer
│   ├── models.py          # RunRequest, RunResult, Task, Evidence
│   ├── rag_store.py       # Milvus Lite RAG build/retrieve
│   ├── system.py          # ThreeAgentSystem execution loop
│   ├── tracer.py          # Langfuse tracing integration
│   └── utils.py           # Text/keyword helpers
├── data/
│   ├── rag/               # Internal policy documents
│   │   ├── escalation_matrix.md
│   │   ├── payment_disputes.md
│   │   ├── privacy_requests.md
│   │   └── refunds_policy.md
│   └── rag_eval.py        # Retrieval evaluation queries
├── log/                   # Run outputs (JSON/TXT)
├── vector_db/             # Milvus Lite database artifacts
├── main.py                # CLI runner
├── run_milvius.py         # Build/reuse/evaluate RAG data
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy environment variables:

```bash
cp .env.example .env
```

3. Build or reuse the RAG index:

```bash
python run_milvius.py
```

4. Run the CLI workflow:

```bash
python main.py --goal "customer is escalating for refund that is not being booked"
```

## CLI Usage

Run interactively:

```bash
python main.py
```

Available options:

```bash
--goal             High-level complaint or support/legal question.
--case-context     Optional extra context.
--provider         LLM provider: openai or gemini.
--embedding-provider Embedding provider: openai or gemini.
--public-url       One or more public URLs for web evidence.
--json             Print full JSON output.
```

## Provider Switching

OpenAI:

```bash
python run_milvius.py --embedding-provider openai
python main.py --provider openai --embedding-provider openai --goal "guest cancelled but was still charged"
```

Gemini:

```bash
python run_milvius.py --embedding-provider gemini
python main.py --provider gemini --embedding-provider gemini --goal "guest cancelled but was still charged"
```

## RAG Build and Evaluation

- `run_milvius.py` builds or reuses the Milvus Lite index from `data/rag/`.
- Documents are chunked into ~700-character windows with sentence overlap.
- Embeddings are stored and retrieved with cosine similarity.
- Evaluation uses `data/rag_eval.py` to measure hit rate and MRR on sample queries.

Example evaluation cases:
- customer cancelled inside free cancellation window but was still charged
- property marked booking as no-show after guest says they cancelled
- chargeback dispute for duplicate property charge
- customer requests deletion of personal data

## Internal Architecture

### Planner

`agent_system/agents.py` defines `PlannerAgent`, which:
- creates a fallback plan from the goal and context
- normalizes model output into exactly five tasks
- generates action plan, next steps, and final answer using evidence summaries

### RAG Retriever

`RagRetrieverAgent` retrieves internal policy evidence and summarizes it with the LLM.

### Web Searcher

`WebSearcherAgent` collects public evidence from Booking.com via:
- `SerpAPI` when configured
- `BeautifulSoup` scraping otherwise

Evidence is ranked by keyword overlap and distilled into summary bullets.

## Context Management

This repository avoids prompt overflow by:
- generating `web_keywords` instead of full page text
- retrieving only top-k evidence chunks
- summarizing evidence before feeding it back into planner prompts
- keeping each task prompt focused on a single responsibility

## Notes

- No heavy agent framework is used; the implementation is simple Python orchestration.
- The RAG builder is separate from the live runtime.
- SerpAPI is optional — fallback is BeautifulSoup scraping of provided public URLs.
- `log/` stores output artifacts per run.

## Evaluation Setup

![LLM as Judge Evaluation](image/LLM_as_Judge.png)
![Trace Evaluation](image/Trace_evaluation.png)

Used Langfuse LLM-as-a-Judge evaluators to measure the quality of agent outputs at different stages of execution.
This helps us validate not only the final response, but also the intermediate steps such as planning, retrieval, and web search.

### Evaluators Used

**Helpfulness**  
Measures whether the generated response is useful, clear, and aligned with the user’s request.  
A higher score indicates that the response is actionable and answers the task effectively.

**Hallucination**  
Measures whether the response contains unsupported, fabricated, or misleading information.  
A lower score is better here, since it indicates the model stayed grounded and avoided making things up.

**Relevance**  
Measures whether the retrieved or generated content is relevant to the current task or query.  
A higher score indicates that the agent is selecting information that is closely related to the user’s intent.

### How the Evaluation Is Applied

The evaluators are attached to different parts of the agent trace:

- Planner output is checked for Helpfulness and Hallucination
- Retriever results are checked for Relevance
- Web search results are also checked for Relevance

This lets us evaluate the system at a component level instead of only judging the final answer.  
For example:

A high helpfulness score means the planner is producing useful next steps  
A low hallucination score means the planner output is grounded  
A high relevance score means the retriever and web searcher are fetching useful supporting information

### Why This Matters

By evaluating each stage independently, we can identify where the pipeline performs well and where it needs improvement.  
This makes debugging easier and provides a more reliable way to improve overall agent quality.

**Typical benefits of this setup:**
- Detect unhelpful planning steps early
- Catch hallucinated content before it reaches the final answer
- Verify that retrieval and search components are returning contextually relevant information
- Improve trust and consistency in the end-to-end agent workflow

### Example Interpretation

A trace with:

- Helpfulness: 1.00
- Hallucination: 0.00
- Relevance: 1.00

can be interpreted as:

- the planner produced a useful response,
- the output did not contain unsupported information,
- and the retrieved/search results were relevant to the task.
