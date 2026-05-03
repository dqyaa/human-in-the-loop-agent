"""
Human-in-the-Loop Agent — FastAPI Backend
==========================================
REST API for the complaint handling workflow.

Endpoints:
    POST /complaints/           — Submit a new complaint (starts the workflow)
    GET  /complaints/pending    — List complaints waiting for human review
    POST /complaints/{id}/decide — Submit approve/reject/modify decision
    GET  /complaints/{id}       — Get full state of a complaint
    GET  /stats                 — Dashboard statistics
    GET  /health                — Health check

The workflow lifecycle:
    1. Client POSTs a complaint → agent runs until REVIEW node → pauses
    2. Supervisor GETs /pending → sees complaint + draft
    3. Supervisor POSTs /decide → agent resumes → response sent → closed

Run:
    uvicorn api.main:app --reload --port 8000
"""

import uuid
import logging
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langgraph.types import Command

logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────

_graph = None
_thread_manager = None
_active_threads: dict[str, str] = {}   # complaint_id → thread_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise graph and checkpointer on startup."""
    global _graph, _thread_manager

    from agent.graph import build_graph
    from agent.checkpointer import get_checkpointer, ThreadManager

    checkpointer = get_checkpointer("complaints.db")
    _thread_manager = ThreadManager(checkpointer, "complaints.db")
    _graph = build_graph(checkpointer=checkpointer)

    logger.info("✅ Complaint handling agent ready")
    yield


app = FastAPI(
    title="Human-in-the-Loop Agent API",
    description=(
        "Customer complaint handling agent with human supervisor approval.\n\n"
        "**Flow:** Submit complaint → Agent drafts response → "
        "Supervisor reviews → Agent sends approved response\n\n"
        "Built by Aliya Alias | github.com/aliyaalias19"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ───────────────────────────────────────────────────

class ComplaintRequest(BaseModel):
    customer_name: str
    customer_email: str
    complaint_text: str
    channel: str = "email"


class ComplaintResponse(BaseModel):
    complaint_id: str
    thread_id: str
    status: str
    message: str


class ReviewDecision(BaseModel):
    decision: str               # "approve" | "reject" | "modify"
    notes: Optional[str] = None
    modified_response: Optional[str] = None
    reviewed_by: str = "supervisor"


class PendingReview(BaseModel):
    thread_id: str
    complaint_id: str
    customer_name: str
    customer_email: str
    complaint_text: str
    category: str
    urgency: str
    draft_response: str
    draft_reasoning: str
    draft_version: int
    revision_count: int
    created_at: str


# ── Helper: run agent in background ──────────────────────────────────────────

def _run_agent_thread(thread_id: str, state_dict: dict):
    """Run the agent graph synchronously in a background thread."""
    global _graph, _thread_manager

    config = {"configurable": {"thread_id": thread_id}}

    try:
        from agent.graph import ComplaintState
        initial_state = ComplaintState(**state_dict)

        # Run until interrupt (REVIEW node)
        for chunk in _graph.stream(initial_state, config=config, stream_mode="values"):
            pass   # Consume stream — graph pauses at interrupt()

        logger.info(f"[{thread_id}] Agent paused at REVIEW node")

    except Exception as e:
        logger.error(f"[{thread_id}] Agent error: {e}")
        if _thread_manager:
            _thread_manager.update_thread_status(thread_id, "error")


def _resume_agent_thread(thread_id: str, decision_payload: dict):
    """Resume a paused agent with human decision."""
    global _graph, _thread_manager

    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Resume with human decision
        for chunk in _graph.stream(
            Command(resume=decision_payload),
            config=config,
            stream_mode="values"
        ):
            pass

        if _thread_manager:
            _thread_manager.update_thread_status(thread_id, "resolved")
        logger.info(f"[{thread_id}] Agent completed after human decision")

    except Exception as e:
        logger.error(f"[{thread_id}] Resume error: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "complaint-handling-hitl",
        "checkpointer": "sqlite" if _graph else "not_initialised",
    }


@app.post("/complaints/", response_model=ComplaintResponse)
async def submit_complaint(
    request: ComplaintRequest,
    background_tasks: BackgroundTasks,
):
    """
    Submit a new customer complaint.
    The agent starts immediately, drafts a response, then pauses for human review.
    Returns immediately — agent runs in background.
    """
    complaint_id = str(uuid.uuid4())[:8]
    thread_id = f"complaint_{complaint_id}_{uuid.uuid4().hex[:6]}"

    state_dict = {
        "complaint_id": complaint_id,
        "customer_name": request.customer_name,
        "customer_email": request.customer_email,
        "complaint_text": request.complaint_text,
        "channel": request.channel,
    }

    _active_threads[complaint_id] = thread_id

    if _thread_manager:
        _thread_manager.register_thread(
            thread_id=thread_id,
            complaint_id=complaint_id,
            customer=request.customer_name,
            category="pending",
            urgency="pending",
        )

    # Run agent in background thread (pauses at REVIEW node)
    background_tasks.add_task(_run_agent_thread, thread_id, state_dict)

    return ComplaintResponse(
        complaint_id=complaint_id,
        thread_id=thread_id,
        status="processing",
        message=(
            "Complaint received. Agent is drafting a response. "
            "Check /complaints/pending in a few seconds for supervisor review."
        ),
    )


@app.get("/complaints/pending", response_model=list[dict])
async def get_pending_reviews():
    """
    List all complaints currently waiting for supervisor review.
    Returns complaints sorted by urgency (critical first).
    """
    if not _thread_manager:
        raise HTTPException(503, "Agent not initialised")

    pending = _thread_manager.get_pending_reviews()

    # Enrich with current graph state
    result = []
    for thread in pending:
        thread_id = thread["thread_id"]
        config = {"configurable": {"thread_id": thread_id}}
        try:
            state = _graph.get_state(config)
            if state and state.values:
                values = state.values
                result.append({
                    **thread,
                    "draft_response": values.get("draft_response", ""),
                    "draft_reasoning": values.get("draft_reasoning", ""),
                    "draft_version": values.get("draft_version", 1),
                    "category": values.get("category", "Unknown"),
                    "urgency": values.get("urgency", "unknown"),
                    "revision_count": values.get("revision_count", 0),
                    "complaint_text": values.get("complaint_text", ""),
                    "customer_email": values.get("customer_email", ""),
                })
        except Exception as e:
            logger.debug(f"State fetch error for {thread_id}: {e}")
            result.append(thread)

    return result


@app.post("/complaints/{complaint_id}/decide")
async def submit_decision(
    complaint_id: str,
    decision: ReviewDecision,
    background_tasks: BackgroundTasks,
):
    """
    Submit supervisor's decision on a pending complaint.

    Decision options:
    - **approve**: Send the draft as-is
    - **reject**: Discard draft, agent will regenerate (provide notes with reason)
    - **modify**: Send the supervisor's edited version (provide modified_response)
    """
    thread_id = _active_threads.get(complaint_id)
    if not thread_id:
        raise HTTPException(404, f"Complaint {complaint_id} not found or already resolved")

    decision_payload = {
        "decision": decision.decision,
        "notes": decision.notes or "",
        "modified_response": decision.modified_response,
        "reviewed_by": decision.reviewed_by,
    }

    if _thread_manager:
        status = "resolved" if decision.decision != "reject" else "pending_review"
        _thread_manager.update_thread_status(thread_id, status)

    # Resume the paused agent in background
    background_tasks.add_task(_resume_agent_thread, thread_id, decision_payload)

    action_map = {
        "approve": "Response approved and will be sent to customer",
        "reject": "Draft rejected — agent is regenerating response",
        "modify": "Modified response will be sent to customer",
    }

    return {
        "complaint_id": complaint_id,
        "thread_id": thread_id,
        "decision": decision.decision,
        "message": action_map.get(decision.decision, "Decision recorded"),
    }


@app.get("/complaints/{complaint_id}")
async def get_complaint(complaint_id: str):
    """Get full state of a complaint workflow."""
    thread_id = _active_threads.get(complaint_id)
    if not thread_id:
        raise HTTPException(404, f"Complaint {complaint_id} not found")

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = _graph.get_state(config)
        return {
            "complaint_id": complaint_id,
            "thread_id": thread_id,
            "state": state.values if state else {},
            "next_node": list(state.next) if state else [],
        }
    except Exception as e:
        raise HTTPException(500, f"State fetch error: {e}")


@app.get("/stats")
async def get_stats():
    """Dashboard statistics."""
    stats = _thread_manager.get_stats() if _thread_manager else {}
    pending = _thread_manager.get_pending_reviews() if _thread_manager else []

    return {
        "by_status": stats,
        "pending_count": len(pending),
        "high_urgency_pending": sum(
            1 for p in pending
            if p.get("urgency") in ("high", "critical")
        ),
        "active_threads": len(_active_threads),
    }


@app.get("/")
async def root():
    return {
        "message": "Human-in-the-Loop Complaint Handling Agent",
        "docs": "/docs",
        "flow": {
            "1_submit": "POST /complaints/",
            "2_review": "GET /complaints/pending",
            "3_decide": "POST /complaints/{id}/decide",
            "4_status": "GET /complaints/{id}",
        },
    }
