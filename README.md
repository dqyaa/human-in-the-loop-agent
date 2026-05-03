# 🤝 Human-in-the-Loop Agent

> A LangGraph agent that **pauses for human approval** before sending customer responses. The agent drafts — the human decides — the agent executes.

[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-HITL-blue)](https://langchain.com/langgraph)
[![License](https://img.shields.io/badge/License-MIT-orange)](LICENSE)

---

## The Problem

AI agents act without asking. An agent that sends the wrong response to a frustrated customer, commits a breaking change, or triggers a $2,000 API bill does not stop to ask for permission. It just acts.

> *"Your autonomous coding agent refactored the authentication module while you were in a meeting. It looked right to the LLM. It broke production."*

## The Solution: Plan → Review → Execute

```
Customer Complaint
       ↓
  [INTAKE]    — Classify category, urgency, sentiment
       ↓
  [DRAFT]     — LLM generates response draft
       ↓
  [REVIEW]    — ⏸ HARD PAUSE via interrupt()
       |          State saved to SQLite
       |          Supervisor sees: complaint + draft + reasoning
       ↓
  Supervisor decision:
  ├── approve ──→ [EXECUTE] → Response sent ✉
  ├── reject  ──→ [DRAFT]   → Regenerate (max 3 revisions)
  └── modify  ──→ [EXECUTE] → Edited response sent ✉
       ↓
  [CLOSE]     — Audit log recorded
```

**The pause can last hours or days.** State is checkpointed to SQLite, so restarts don't lose pending approvals.

---

## Quick Start

```bash
git clone https://github.com/aliyaalias19/human-in-the-loop-agent
cd human-in-the-loop-agent
pip install -r requirements.txt

# Terminal demo (no API needed)
python quickstart.py
```

### Full Stack (API + Dashboard)

```bash
# Terminal 1: Start the API
uvicorn api.main:app --port 8000

# Terminal 2: Start the dashboard
python demo/app.py
# Open: http://localhost:7860
```

### API Usage

```bash
# Submit a complaint
curl -X POST http://localhost:8000/complaints/ \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name": "Aishah Rahman",
    "customer_email": "aishah@example.com",
    "complaint_text": "Refund not received after 3 weeks. Order ORD-2024-88123.",
    "channel": "email"
  }'

# See pending reviews
curl http://localhost:8000/complaints/pending

# Submit decision
curl -X POST http://localhost:8000/complaints/{complaint_id}/decide \
  -H "Content-Type: application/json" \
  -d '{
    "decision": "approve",
    "reviewed_by": "supervisor_ali"
  }'
```

---

## Key LangGraph Concepts

### interrupt() — The Hard Pause

```python
def review_node(state: ComplaintState) -> dict:
    # Execution STOPS here. State is checkpointed.
    # The payload is returned to the caller (API/UI).
    human_decision = interrupt({
        "complaint_text": state.complaint_text,
        "draft_response": state.draft_response,
        "category": state.category,
        "urgency": state.urgency,
    })

    # Execution RESUMES here after Command(resume=...) is called.
    decision = human_decision["decision"]  # "approve" | "reject" | "modify"
    return {"review_decision": decision, ...}
```

### Command(resume=data) — The Resumption

```python
# Human makes decision → API receives it → Graph resumes
for event in graph.stream(
    Command(resume={
        "decision": "approve",
        "reviewed_by": "supervisor_ali",
    }),
    config={"configurable": {"thread_id": thread_id}},
):
    pass
```

### SqliteSaver — Persistence That Survives Restarts

```python
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("complaints.db")
checkpointer = SqliteSaver(conn)
graph = build_graph(checkpointer=checkpointer)

# State is saved after every node.
# If the server restarts while a complaint is pending:
# - The thread_id still exists in SQLite
# - The supervisor can still submit a decision
# - The graph resumes from the last checkpoint
```

---

## Graph Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  ComplaintState (TypedDict)                  │
│  complaint_id │ customer │ complaint_text │ category        │
│  urgency │ sentiment │ draft_response │ review_decision     │
│  final_response │ node_history │ revision_count            │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
           INTAKE           DRAFT           REVIEW ⏸
           (fast)           (LLM)          (interrupt)
              │               ↑               │
              └───────────────┘   approve/modify → EXECUTE → CLOSE
                                  reject ──────→ DRAFT (loop, max 3)
                                  circuit break → CLOSE
```

---

## Production Considerations

### TTL-based Expiry
Threads not reviewed within 24h are auto-expired:
```python
manager.expire_old_threads(ttl_hours=24)
```

### Thread ID Isolation
Never reuse thread IDs — one per complaint, one per user session.

### Interrupt on Irreversible Actions Only
HITL is a cost: human time, latency, coordination. Reserve it for:
- Customer-facing communications
- Irreversible database writes
- Financial transactions
- High-blast-radius actions

### Upgrade Checkpointer for Production
```python
# Development (in-memory — state lost on restart)
from langgraph.checkpoint.memory import MemorySaver

# Production (SQLite — survives restarts)
from langgraph.checkpoint.sqlite import SqliteSaver

# Production scale (PostgreSQL — distributed)
from langgraph.checkpoint.postgres import PostgresSaver
```

---

## File Structure

```
human-in-the-loop-agent/
│
├── agent/
│   ├── graph.py          ← LangGraph nodes, state, edges, interrupt()
│   └── checkpointer.py   ← SQLite persistence, TTL expiry, thread management
│
├── api/
│   └── main.py           ← FastAPI: submit complaint, review, decide, stats
│
├── demo/
│   └── app.py            ← Gradio supervisor dashboard
│
├── quickstart.py         ← Terminal demo (no API needed)
└── requirements.txt
```

---

## Citation

```bibtex
@misc{alias2026hitl,
  title  = {Human-in-the-Loop Agent: LangGraph HITL with interrupt() and SQLite Checkpointing},
  author = {Alias, Aliya},
  year   = {2026},
  url    = {https://github.com/aliyaalias19/human-in-the-loop-agent}
}
```

---

## 👤 About

Built by **Aliya Alias** — AI Engineer, Kuala Lumpur.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-aliyaalias-blue)](https://linkedin.com/in/aliyaalias)
[![GitHub](https://img.shields.io/badge/GitHub-aliyaalias19-black)](https://github.com/aliyaalias19)

*MIT License.*
