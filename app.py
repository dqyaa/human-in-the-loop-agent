"""
Supervisor Dashboard — Gradio UI
==================================
The human review interface for the complaint handling agent.
Supervisors see pending complaints, review draft responses,
and make approve/reject/modify decisions.

Run:
    python demo/app.py
    # Open: http://localhost:7860

Note: Requires the FastAPI backend running on port 8000:
    uvicorn api.main:app --port 8000
"""

import sys
import json
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

API_BASE = "http://localhost:8000"

URGENCY_COLORS = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🟢",
}


def get_pending() -> tuple[list, str]:
    """Fetch pending complaints from API."""
    try:
        r = requests.get(f"{API_BASE}/complaints/pending", timeout=5)
        items = r.json() if r.status_code == 200 else []
        count = len(items)
        status = f"✅ {count} complaint(s) pending review" if count else "✅ No pending reviews"
        return items, status
    except Exception as e:
        return [], f"❌ API not available: {e}"


def submit_complaint(name: str, email: str, complaint: str, channel: str) -> str:
    """Submit a new complaint via API."""
    if not all([name.strip(), email.strip(), complaint.strip()]):
        return "⚠️ Please fill in all fields."
    try:
        r = requests.post(
            f"{API_BASE}/complaints/",
            json={
                "customer_name": name,
                "customer_email": email,
                "complaint_text": complaint,
                "channel": channel,
            },
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            return (
                f"✅ Complaint submitted!\n"
                f"   Complaint ID: **{data['complaint_id']}**\n"
                f"   Thread ID: `{data['thread_id']}`\n\n"
                f"The agent is now classifying and drafting a response.\n"
                f"Check the **Review Queue** tab in ~3 seconds."
            )
        return f"❌ Error: {r.text[:200]}"
    except Exception as e:
        return f"❌ API error: {e}"


def load_review_queue() -> tuple[list[list], str]:
    """Load pending reviews for the table."""
    items, status = get_pending()
    rows = []
    for item in items:
        urgency = item.get("urgency", "unknown")
        icon = URGENCY_COLORS.get(urgency, "⚪")
        rows.append([
            item.get("complaint_id", ""),
            item.get("customer", item.get("customer_name", "")),
            item.get("category", "Unknown"),
            f"{icon} {urgency}",
            item.get("draft_version", 1),
            item.get("created_at", "")[:16],
        ])
    return rows, status


def load_complaint_detail(complaint_id: str) -> tuple[str, str, str, str]:
    """Load full detail for a specific complaint."""
    items, _ = get_pending()
    for item in items:
        if item.get("complaint_id") == complaint_id or item.get("thread_id", "").endswith(complaint_id):
            complaint_text = item.get("complaint_text", "")
            category = item.get("category", "Unknown")
            urgency = item.get("urgency", "unknown")
            draft = item.get("draft_response", "")
            reasoning = item.get("draft_reasoning", "")
            revision = item.get("revision_count", 0)

            detail = (
                f"**Customer:** {item.get('customer', item.get('customer_name', ''))}\n"
                f"**Email:** {item.get('customer_email', '')}\n"
                f"**Category:** {category} | **Urgency:** {URGENCY_COLORS.get(urgency, '')} {urgency}\n"
                f"**Draft Version:** {item.get('draft_version', 1)} | **Revisions:** {revision}\n"
                f"{'─'*40}\n"
                f"**Complaint:**\n{complaint_text}"
            )

            agent_info = f"**Agent Reasoning:** {reasoning}"

            return detail, draft, agent_info, complaint_id

    return "No complaint found with that ID. Refresh the queue first.", "", "", ""


def submit_decision(
    complaint_id: str,
    decision: str,
    notes: str,
    modified_response: str,
    supervisor: str,
) -> str:
    """Submit approval decision to the API."""
    if not complaint_id.strip():
        return "⚠️ No complaint selected. Click a row in the queue first."
    if not decision:
        return "⚠️ Please select a decision (Approve / Reject / Modify)."
    if decision == "modify" and not modified_response.strip():
        return "⚠️ You selected 'Modify' — please provide your edited response."
    if decision == "reject" and not notes.strip():
        return "⚠️ Please provide a reason for rejection so the agent can improve the draft."

    payload = {
        "decision": decision.lower(),
        "notes": notes,
        "modified_response": modified_response if decision == "modify" else None,
        "reviewed_by": supervisor or "supervisor",
    }

    try:
        r = requests.post(
            f"{API_BASE}/complaints/{complaint_id}/decide",
            json=payload,
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            action = {
                "approve": "✅ Response APPROVED and will be sent to customer",
                "reject": "🔄 Draft REJECTED — agent is regenerating with your feedback",
                "modify": "✏️ Modified response ACCEPTED and will be sent to customer",
            }.get(decision.lower(), "Decision recorded")
            return f"{action}\n\nComplaint: {data.get('complaint_id')}"
        return f"❌ Error: {r.text[:200]}"
    except Exception as e:
        return f"❌ API error: {e}"


def get_stats() -> str:
    """Get dashboard stats."""
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return (
                f"📊 **Dashboard Stats**\n"
                f"Pending review: **{data.get('pending_count', 0)}**\n"
                f"High urgency: **{data.get('high_urgency_pending', 0)}**\n"
                f"Active threads: **{data.get('active_threads', 0)}**\n"
                f"By status: {json.dumps(data.get('by_status', {}), indent=2)}"
            )
        return "Stats unavailable"
    except Exception as e:
        return f"❌ API not available: {e}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Complaint Handling — HITL Supervisor Dashboard",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.HTML("""
        <div style="text-align:center; padding:16px 0 8px;">
            <h1>🤝 Human-in-the-Loop Agent</h1>
            <p>Customer complaint handling with supervisor approval gate.</p>
            <p style="font-size:0.85em; color:#888;">
                Agent drafts response → Supervisor approves → Customer receives response
            </p>
        </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Submit Complaint ───────────────────────────────────────────
        with gr.Tab("📨 Submit Complaint"):
            gr.Markdown("### Simulate a customer submitting a complaint")

            with gr.Row():
                cust_name = gr.Textbox(label="Customer Name", value="Aliya Alias")
                cust_email = gr.Textbox(label="Customer Email", value="aliya@example.com")
                channel = gr.Dropdown(
                    choices=["email", "chat", "phone", "social"],
                    value="email", label="Channel"
                )

            complaint_text = gr.Textbox(
                label="Complaint Text",
                lines=5,
                value="I submitted a refund request 3 weeks ago for order #ORD-2024-88123 "
                      "and I still haven't received my money back (RM 450). "
                      "I've contacted support twice and nobody responds. This is absolutely unacceptable!"
            )

            submit_btn = gr.Button("📨 Submit Complaint", variant="primary")
            submit_status = gr.Markdown("Fill in the form and click Submit.")

            gr.HTML("""
                <div style="padding:12px; background:#e0f2fe; border-radius:8px; margin-top:12px;">
                    <b>Try these example complaints:</b>
                    <ul>
                        <li>Refund: "I've been waiting 3 weeks for my RM 450 refund..."</li>
                        <li>Technical: "Your app keeps crashing whenever I try to login..."</li>
                        <li>Billing: "I was charged twice for the same order last month..."</li>
                        <li>Service: "The staff at your KL branch was extremely rude..."</li>
                    </ul>
                </div>
            """)

            submit_btn.click(
                submit_complaint,
                [cust_name, cust_email, complaint_text, channel],
                submit_status,
            )

        # ── Tab 2: Review Queue ───────────────────────────────────────────────
        with gr.Tab("📋 Review Queue"):
            gr.Markdown("### Complaints waiting for your approval")
            gr.Markdown("*Click a row to load the full complaint details below.*")

            refresh_btn = gr.Button("🔄 Refresh Queue", size="sm")
            queue_status = gr.Markdown("")

            queue_table = gr.Dataframe(
                headers=["Complaint ID", "Customer", "Category", "Urgency", "Draft #", "Submitted"],
                datatype=["str", "str", "str", "str", "number", "str"],
                interactive=False,
                label="Pending Reviews",
            )

            gr.Markdown("---")
            gr.Markdown("### 📄 Complaint Detail & Review")

            with gr.Row():
                selected_id = gr.Textbox(
                    label="Complaint ID",
                    placeholder="Enter complaint ID from table above",
                    scale=2,
                )
                load_btn = gr.Button("📂 Load Complaint", scale=1)

            with gr.Row():
                complaint_detail = gr.Markdown("*No complaint loaded*", label="Complaint Info")

            agent_reasoning = gr.Markdown("", label="Agent Reasoning")

            draft_display = gr.Textbox(
                label="📝 Agent Draft Response",
                lines=10,
                interactive=False,
            )

            gr.Markdown("### ✅ Your Decision")

            with gr.Row():
                decision_radio = gr.Radio(
                    choices=["approve", "reject", "modify"],
                    label="Decision",
                    info="approve=send as-is | reject=regenerate | modify=edit and send",
                )
                supervisor_name = gr.Textbox(label="Your Name", value="Supervisor")

            rejection_notes = gr.Textbox(
                label="Notes / Rejection Reason",
                placeholder="Required for reject. Optional for approve/modify.",
                lines=2,
            )
            modified_response = gr.Textbox(
                label="Modified Response (required if 'modify' selected)",
                lines=8,
                placeholder="Edit the agent's draft here if you selected 'modify'...",
            )

            decide_btn = gr.Button("⚡ Submit Decision", variant="primary", size="lg")
            decision_status = gr.Markdown("")

            # Wire up
            refresh_btn.click(load_review_queue, outputs=[queue_table, queue_status])
            load_btn.click(
                load_complaint_detail,
                inputs=[selected_id],
                outputs=[complaint_detail, draft_display, agent_reasoning, selected_id],
            )
            decide_btn.click(
                submit_decision,
                inputs=[selected_id, decision_radio, rejection_notes,
                        modified_response, supervisor_name],
                outputs=decision_status,
            )

        # ── Tab 3: Architecture ────────────────────────────────────────────────
        with gr.Tab("🏗 Architecture"):
            gr.Markdown("""
## How Human-in-the-Loop Works

### The Problem
AI agents act without asking. An agent that sends the wrong response to a
frustrated customer, commits a breaking change, or triggers an expensive API call
does not stop to ask for permission. It just acts.

### The Solution: interrupt() + Checkpointing

```
Customer Complaint
       ↓
  [INTAKE NODE]          ← Classify: category, urgency, sentiment
       ↓
  [DRAFT NODE]           ← LLM generates response draft
       ↓
  [REVIEW NODE]          ← ⏸ HARD PAUSE via interrupt()
       |                    State checkpointed to SQLite
       |                    Human sees: complaint + draft + reasoning
       ↓
  Supervisor Decision:
  ├── approve  ──────────→ [EXECUTE NODE] → Response sent ✉
  ├── reject   ──────────→ [DRAFT NODE]   → Regenerate with feedback
  │   (with reason)                        ↑ Loop up to 3 times
  └── modify   ──────────→ [EXECUTE NODE] → Edited response sent ✉
       ↓
  [CLOSE NODE]           ← Audit log, resolution recorded
```

### Key LangGraph Primitives

| Primitive | What It Does |
|---|---|
| `interrupt(payload)` | Freezes graph, returns payload to caller |
| `Command(resume=data)` | Resumes frozen graph with human's data |
| `SqliteSaver` | Persists ALL state to SQLite (survives restarts) |
| `thread_id` | Unique ID per workflow instance |
| `conditional_edges` | Routes: approve→execute, reject→draft, modify→execute |

### Circuit Breaker
If the agent fails to produce an acceptable draft after 3 revisions,
the circuit breaker routes to CLOSE node directly instead of looping forever.

### Production Considerations
- **TTL**: Threads not reviewed in 24h are auto-expired
- **Thread IDs**: Never reuse thread IDs — one per complaint
- **Checkpointer**: Use PostgreSQL in production (not SQLite)
- **Interrupt on irreversible actions only** — not every step

---
Built by **Aliya Alias** | [GitHub](https://github.com/aliyaalias19/human-in-the-loop-agent)
            """)

        # ── Tab 4: Stats ──────────────────────────────────────────────────────
        with gr.Tab("📊 Stats"):
            stats_btn = gr.Button("🔄 Refresh Stats")
            stats_display = gr.Markdown("Click refresh to load stats.")
            stats_btn.click(get_stats, outputs=stats_display)

    # Auto-refresh queue on load
    demo.load(load_review_queue, outputs=[queue_table, queue_status])


if __name__ == "__main__":
    print("🤝 Starting Human-in-the-Loop Supervisor Dashboard...")
    print("   Open: http://localhost:7860")
    print("   NOTE: FastAPI backend must be running on port 8000")
    print("   Run in another terminal: uvicorn api.main:app --port 8000")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
