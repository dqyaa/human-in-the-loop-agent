"""
Human-in-the-Loop Agent — State & Graph
=========================================
A LangGraph agent that handles customer complaints by:

  1. INTAKE    — Parse and classify the incoming complaint
  2. DRAFT     — Generate a response draft using the LLM
  3. REVIEW    — ⏸ HARD PAUSE — human supervisor reviews and decides
  4. EXECUTE   — Send the approved response (or revise if rejected)
  5. CLOSE     — Log the resolution

The REVIEW node uses interrupt() to completely freeze execution.
State is checkpointed to SQLite so the pause can last hours or days.
When the supervisor responds (approve/reject/modify), execution resumes
exactly where it stopped.

Why customer complaint handling as the demo scenario?
- High stakes: A wrong response damages brand reputation
- Time-sensitive: Needs review but can't block for hours
- Real example of irreversible action: Once sent to customer, it's sent
- Directly relevant to your Aevoco (LLM pipeline) and banking context

Architecture:
    intake → draft → review ──[approve]──→ execute → close
                        │
                        ├──[reject]───→ draft (revise)
                        │
                        └──[modify]───→ execute (with edits)
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Annotated, Literal, Optional
from dataclasses import dataclass

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


# ── State definition ──────────────────────────────────────────────────────────

class ComplaintState(BaseModel):
    """
    Complete state of a complaint handling workflow.
    Every node reads from and writes to this state.
    This is the single source of truth — makes debugging trivial.
    """

    # Input
    complaint_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    customer_name: str = ""
    customer_email: str = ""
    complaint_text: str = ""
    channel: str = "email"           # email | chat | phone | social

    # Classification (set by INTAKE node)
    category: Optional[str] = None   # Refund | Technical | Billing | General
    urgency: Optional[str] = None    # low | medium | high | critical
    sentiment_score: Optional[float] = None   # -1.0 to 1.0
    key_issue: Optional[str] = None

    # Draft response (set by DRAFT node)
    draft_response: Optional[str] = None
    draft_reasoning: Optional[str] = None    # Why the agent chose this approach
    draft_version: int = 0

    # Human review decision (set by REVIEW node / human)
    review_decision: Optional[Literal["approve", "reject", "modify"]] = None
    review_notes: Optional[str] = None       # Supervisor's notes/reason
    modified_response: Optional[str] = None  # If supervisor modified the draft
    reviewed_by: Optional[str] = None        # Supervisor identifier
    reviewed_at: Optional[str] = None

    # Execution (set by EXECUTE node)
    final_response: Optional[str] = None
    sent_at: Optional[str] = None
    execution_success: bool = False

    # History / audit trail
    node_history: list[str] = Field(default_factory=list)
    draft_history: list[dict] = Field(default_factory=list)   # All drafts created
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Control
    max_revisions: int = 3
    revision_count: int = 0
    error: Optional[str] = None


# ── Node functions ────────────────────────────────────────────────────────────

def intake_node(state: ComplaintState) -> dict:
    """
    Parse and classify the incoming complaint.
    No LLM needed — fast regex + keyword classification.
    """
    state.node_history.append("intake")
    text_lower = state.complaint_text.lower()

    # Category classification
    if any(w in text_lower for w in ["refund", "money back", "bayar balik", "return"]):
        category = "Refund"
    elif any(w in text_lower for w in ["not working", "error", "bug", "crash", "technical"]):
        category = "Technical"
    elif any(w in text_lower for w in ["bill", "charge", "payment", "invoice", "bil"]):
        category = "Billing"
    elif any(w in text_lower for w in ["delivery", "shipping", "order", "hantar"]):
        category = "Delivery"
    elif any(w in text_lower for w in ["rude", "staff", "service", "attitude"]):
        category = "Service Quality"
    else:
        category = "General"

    # Urgency
    urgent_words = ["urgent", "immediately", "asap", "emergency", "critical",
                    "segera", "cepat", "sekarang", "disappointed", "unacceptable"]
    high_words = ["frustrated", "angry", "waiting", "weeks", "months", "still"]

    if any(w in text_lower for w in urgent_words):
        urgency = "high"
    elif any(w in text_lower for w in high_words):
        urgency = "medium"
    else:
        urgency = "low"

    # Simple sentiment
    negative_words = ["angry", "frustrated", "terrible", "unacceptable",
                      "disappointed", "disgusted", "worst", "marah", "kecewa"]
    negative_count = sum(1 for w in negative_words if w in text_lower)
    sentiment = max(-1.0, -0.2 * negative_count)

    # Key issue (first sentence)
    sentences = state.complaint_text.split(".")
    key_issue = sentences[0].strip()[:150] if sentences else state.complaint_text[:150]

    logger.info(f"[{state.complaint_id}] Classified: {category}, urgency={urgency}")

    return {
        "category": category,
        "urgency": urgency,
        "sentiment_score": round(sentiment, 2),
        "key_issue": key_issue,
        "node_history": state.node_history,
    }


def draft_node(state: ComplaintState, llm_client=None) -> dict:
    """
    Generate a response draft using the LLM.
    Falls back to a template if no LLM client provided.
    """
    history = state.node_history + ["draft"]
    state.draft_version += 1

    # Build draft using LLM or template
    if llm_client is not None:
        draft, reasoning = _generate_with_llm(state, llm_client)
    else:
        draft, reasoning = _generate_with_template(state)

    # Save to history
    draft_entry = {
        "version": state.draft_version,
        "draft": draft,
        "reasoning": reasoning,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rejection_reason": state.review_notes if state.review_decision == "reject" else None,
    }

    logger.info(f"[{state.complaint_id}] Draft v{state.draft_version} generated")

    return {
        "draft_response": draft,
        "draft_reasoning": reasoning,
        "draft_version": state.draft_version,
        "draft_history": state.draft_history + [draft_entry],
        "node_history": history,
        # Reset review fields for new draft
        "review_decision": None,
        "review_notes": None,
    }


def _generate_with_template(state: ComplaintState) -> tuple[str, str]:
    """Template-based draft generation (no LLM needed)."""
    name = state.customer_name or "Valued Customer"
    category = state.category or "General"
    rejection_context = ""

    if state.review_notes and state.review_decision == "reject":
        rejection_context = f"\n\n[REVISION NOTE: Previous draft was rejected. Reason: {state.review_notes}. Please address this specifically.]"

    templates = {
        "Refund": f"""Dear {name},

Thank you for reaching out to us regarding your refund request. We sincerely apologise for the inconvenience you have experienced.

We have reviewed your case and have initiated the refund process for your account. The refund of {state.key_issue[:50] if state.key_issue else 'the requested amount'} will be processed within 5–7 business days.

We value your patience and understand how important this matter is to you. If you have any further questions, please do not hesitate to contact us.

Warm regards,
Customer Success Team""",

        "Technical": f"""Dear {name},

Thank you for contacting us about the technical issue you are experiencing. We apologise for the disruption this has caused.

Our technical team has been notified and is currently investigating the issue. We expect to have a resolution within 24–48 hours. In the meantime, you may try {f"restarting the application or clearing your browser cache" if "app" in state.complaint_text.lower() else "refreshing the page and trying again"}.

We will send you a follow-up notification once the issue is resolved. Thank you for your patience.

Best regards,
Technical Support Team""",

        "Billing": f"""Dear {name},

Thank you for bringing this billing matter to our attention. We apologise for any confusion or inconvenience this has caused.

We have reviewed your account and the billing concern you raised. Our billing team will investigate this matter and provide you with a detailed response within 2 business days.

If there has been an error on our part, we will rectify it immediately and ensure your account reflects the correct charges.

Sincerely,
Billing Support Team""",

        "General": f"""Dear {name},

Thank you for contacting us. We appreciate you taking the time to share your feedback and apologise that your experience did not meet your expectations.

We take all customer feedback seriously and have escalated your concern to the relevant department. A team member will be in touch with you within 1–2 business days with a full response.

Thank you for your patience and continued support.

Best regards,
Customer Care Team""",
    }

    draft = templates.get(category, templates["General"])
    reasoning = (
        f"Category: {category} | Urgency: {state.urgency} | "
        f"Template selected based on complaint classification. "
        f"Tone adjusted for sentiment score {state.sentiment_score}."
        + rejection_context
    )

    return draft, reasoning


def _generate_with_llm(state: ComplaintState, llm_client) -> tuple[str, str]:
    """LLM-based draft generation."""
    rejection_context = ""
    if state.review_notes and state.review_decision == "reject":
        rejection_context = f"\nIMPORTANT: Previous draft was REJECTED. Reason: '{state.review_notes}'. Fix this specifically."

    prompt = f"""You are a customer service representative for a Malaysian company.
Write a professional, empathetic response to this customer complaint.

Customer: {state.customer_name}
Category: {state.category}
Urgency: {state.urgency}
Complaint: {state.complaint_text}
{rejection_context}

Requirements:
- Professional but warm tone
- Acknowledge the issue specifically
- Provide a concrete next step or resolution
- 2-3 paragraphs maximum
- End with contact details placeholder
- Do NOT make promises you can't keep

Return ONLY the response text, no preamble."""

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        draft = response.choices[0].message.content.strip()
        reasoning = f"LLM-generated | Model: gpt-4o-mini | Urgency: {state.urgency}"
        return draft, reasoning
    except Exception as e:
        logger.error(f"LLM draft failed: {e}, falling back to template")
        return _generate_with_template(state)


def review_node(state: ComplaintState) -> dict:
    """
    ⏸ HARD PAUSE — Human supervisor reviews the draft response.

    This is the core of the HITL pattern.
    interrupt() freezes the graph completely.
    State is checkpointed to SQLite.
    Execution resumes only when Command(resume=decision) is called.

    The human sees:
    - The original complaint
    - The draft response
    - The agent's reasoning
    - Options: approve | reject (with reason) | modify

    Decision flow after resume:
    - "approve"  → execute_node (send as-is)
    - "reject"   → draft_node (revise with reason)
    - "modify"   → execute_node (send with human edits)
    """
    history = state.node_history + ["review"]

    logger.info(f"[{state.complaint_id}] ⏸ PAUSING for human review (v{state.draft_version})")

    # This is the interrupt — execution stops here.
    # The payload is presented to the human for review.
    human_decision = interrupt({
        "type": "complaint_review",
        "complaint_id": state.complaint_id,
        "customer_name": state.customer_name,
        "customer_email": state.customer_email,
        "complaint_text": state.complaint_text,
        "category": state.category,
        "urgency": state.urgency,
        "draft_response": state.draft_response,
        "draft_reasoning": state.draft_reasoning,
        "draft_version": state.draft_version,
        "revision_count": state.revision_count,
        "instructions": (
            "Review the draft response above. Choose:\n"
            "  approve — Send as written\n"
            "  reject  — Discard and regenerate (provide reason in 'notes')\n"
            "  modify  — Edit the draft and send your version"
        ),
    })

    # Execution resumes here after human provides decision
    decision = human_decision.get("decision", "reject")
    notes = human_decision.get("notes", "")
    modified = human_decision.get("modified_response")
    reviewed_by = human_decision.get("reviewed_by", "supervisor")

    logger.info(f"[{state.complaint_id}] ▶ Resumed: decision={decision}")

    revision_count = state.revision_count
    if decision == "reject":
        revision_count += 1

    return {
        "review_decision": decision,
        "review_notes": notes,
        "modified_response": modified,
        "reviewed_by": reviewed_by,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "revision_count": revision_count,
        "node_history": history,
    }


def execute_node(state: ComplaintState) -> dict:
    """
    Send the approved/modified response.
    In production: send via email API, ticketing system, or CRM.
    Here: log it and mark as sent.
    """
    history = state.node_history + ["execute"]

    # Use modified response if supervisor edited it, otherwise use draft
    final = state.modified_response or state.draft_response or ""

    # Simulate sending (replace with real email/CRM API in production)
    logger.info(
        f"[{state.complaint_id}] ✉ SENDING response to {state.customer_email}\n"
        f"{'─'*50}\n{final}\n{'─'*50}"
    )

    return {
        "final_response": final,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "execution_success": True,
        "node_history": history,
    }


def close_node(state: ComplaintState) -> dict:
    """Log resolution and close the complaint."""
    history = state.node_history + ["close"]

    resolution = {
        "complaint_id": state.complaint_id,
        "status": "resolved",
        "category": state.category,
        "urgency": state.urgency,
        "drafts_created": state.draft_version,
        "review_decision": state.review_decision,
        "reviewed_by": state.reviewed_by,
        "total_time": "calculated_from_timestamps",
        "sent": state.execution_success,
    }

    logger.info(f"[{state.complaint_id}] ✅ CLOSED: {json.dumps(resolution, indent=2)}")

    return {"node_history": history}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_review(state: ComplaintState) -> Literal["execute", "draft", "close"]:
    """
    Conditional routing based on human decision.

    approve → execute (send the draft as-is)
    modify  → execute (send the human's modified version)
    reject  → draft   (regenerate with feedback) — but only up to max_revisions
    """
    decision = state.review_decision

    if decision in ("approve", "modify"):
        return "execute"
    elif decision == "reject":
        if state.revision_count >= state.max_revisions:
            # Circuit breaker — max revisions reached
            logger.warning(
                f"[{state.complaint_id}] Max revisions ({state.max_revisions}) reached. "
                f"Escalating to close."
            )
            return "close"
        return "draft"
    else:
        # Fallback
        return "close"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(checkpointer=None, llm_client=None):
    """
    Build the complaint handling LangGraph.

    Args:
        checkpointer: LangGraph checkpointer (SQLite or memory).
                      REQUIRED for interrupt() to work.
        llm_client:   Optional OpenAI/Anthropic client for LLM drafts.
                      If None, uses template-based generation.
    """
    from functools import partial

    # Bind LLM client to draft node
    _draft = partial(draft_node, llm_client=llm_client)

    builder = StateGraph(ComplaintState)

    # Add nodes
    builder.add_node("intake", intake_node)
    builder.add_node("draft", _draft)
    builder.add_node("review", review_node)
    builder.add_node("execute", execute_node)
    builder.add_node("close", close_node)

    # Add edges
    builder.add_edge(START, "intake")
    builder.add_edge("intake", "draft")
    builder.add_edge("draft", "review")

    # Conditional routing after review
    builder.add_conditional_edges(
        "review",
        route_after_review,
        {
            "execute": "execute",
            "draft": "draft",    # Loop back for revision
            "close": "close",    # Circuit breaker
        }
    )

    builder.add_edge("execute", "close")
    builder.add_edge("close", END)

    # Compile with checkpointer
    # The checkpointer is REQUIRED for interrupt() to work.
    # Without it, the graph cannot pause and resume.
    graph = builder.compile(checkpointer=checkpointer or MemorySaver())

    return graph
