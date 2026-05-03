"""
Quick Start — Human-in-the-Loop Agent
=======================================
Run this to see the full HITL workflow in your terminal.
No API key needed — uses template-based draft generation.

Usage:
    python quickstart.py

What happens:
    1. Submits 3 sample complaints
    2. Agent classifies and drafts responses
    3. You review each draft in terminal
    4. You approve, reject, or modify
    5. Agent sends the final response
"""

import sys
import time
import json
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("🤝 Human-in-the-Loop Agent — Quick Start")
print("=" * 60)

# ── Step 1: Check imports ─────────────────────────────────────────────────────
print("\n📦 Checking imports...")
try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.types import Command
    from langgraph.checkpoint.memory import MemorySaver
    from agent.graph import build_graph, ComplaintState
    from agent.checkpointer import get_checkpointer, ThreadManager
    print("   ✅ LangGraph imported successfully")
except ImportError as e:
    print(f"   ❌ {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# ── Step 2: Build graph ───────────────────────────────────────────────────────
print("\n🔨 Building complaint handling graph...")
checkpointer = MemorySaver()   # In-memory for quickstart (no persistence)
graph = build_graph(checkpointer=checkpointer)
print("   ✅ Graph compiled with nodes: intake → draft → review → execute → close")
print("   ✅ interrupt() configured at REVIEW node")
print("   ✅ SQLite checkpointer ready (use get_checkpointer() for persistence)")

# ── Step 3: Sample complaints ──────────────────────────────────────────────────
SAMPLE_COMPLAINTS = [
    {
        "customer_name": "Aishah binti Rahman",
        "customer_email": "aishah@example.com",
        "complaint_text": (
            "I submitted a refund request 3 weeks ago for order #ORD-2024-88123 "
            "and I still haven't received my money back (RM 450). "
            "I've contacted support twice and nobody responds. This is absolutely unacceptable!"
        ),
        "channel": "email",
    },
    {
        "customer_name": "Raj Kumar",
        "customer_email": "raj.kumar@example.com",
        "complaint_text": (
            "Your app keeps crashing every time I try to open it on my iPhone. "
            "I've reinstalled it 3 times and the problem persists. "
            "I have an important meeting tomorrow and need this working urgently."
        ),
        "channel": "chat",
    },
]

print(f"\n📬 Processing {len(SAMPLE_COMPLAINTS)} sample complaint(s)...")


def run_complaint_with_terminal_review(complaint: dict, thread_id: str) -> None:
    """Run a single complaint through the full HITL workflow."""
    config = {"configurable": {"thread_id": thread_id}}

    state = ComplaintState(**complaint)

    print(f"\n{'='*60}")
    print(f"📨 COMPLAINT FROM: {complaint['customer_name']}")
    print(f"   Channel: {complaint['channel']}")
    print(f"   Text: {complaint['complaint_text'][:100]}...")
    print(f"{'='*60}")

    # Run until interrupt (REVIEW node)
    print("\n🤖 Agent running: intake → classify → draft...", end="", flush=True)

    interrupted_state = None
    interrupt_payload = None

    for event in graph.stream(state, config=config, stream_mode="values"):
        print(".", end="", flush=True)

    # Check if we hit an interrupt
    graph_state = graph.get_state(config)
    if graph_state and graph_state.next:
        interrupted_state = graph_state.values
        # Try to get interrupt payload from the state
        print("\n\n⏸  PAUSED AT REVIEW NODE — Human approval required")
    else:
        print("\n✅ Completed without interrupt")
        return

    # Show the draft for review
    draft = interrupted_state.get("draft_response", "")
    category = interrupted_state.get("category", "Unknown")
    urgency = interrupted_state.get("urgency", "unknown")
    reasoning = interrupted_state.get("draft_reasoning", "")

    print(f"\n📊 Classification:")
    print(f"   Category : {category}")
    print(f"   Urgency  : {urgency}")
    print(f"   Sentiment: {interrupted_state.get('sentiment_score', 0):.2f}")

    print(f"\n📝 Agent Draft Response (v{interrupted_state.get('draft_version', 1)}):")
    print("─" * 50)
    for line in draft.split("\n"):
        print(f"  {line}")
    print("─" * 50)

    print(f"\n🧠 Agent Reasoning: {reasoning[:150]}...")

    # Prompt for decision
    print(f"\n✅ Your Decision:")
    print("   [1] Approve — Send as written")
    print("   [2] Reject  — Regenerate with your feedback")
    print("   [3] Modify  — Use your edited version")

    try:
        choice = input("\n   Enter choice (1/2/3) [default=1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    if choice == "2":
        try:
            reason = input("   Rejection reason: ").strip() or "Please be more specific and apologetic."
        except (EOFError, KeyboardInterrupt):
            reason = "Please be more specific."
        decision_payload = {"decision": "reject", "notes": reason, "reviewed_by": "terminal_user"}
        print(f"\n❌ REJECTED — Agent will regenerate with reason: '{reason}'")
    elif choice == "3":
        print("   Enter your modified response (press Enter twice when done):")
        lines = []
        try:
            while True:
                line = input("   ")
                if not line:
                    break
                lines.append(line)
        except (EOFError, KeyboardInterrupt):
            lines = [draft]
        modified = "\n".join(lines) if lines else draft
        decision_payload = {
            "decision": "modify",
            "modified_response": modified,
            "reviewed_by": "terminal_user",
        }
        print(f"\n✏️  MODIFIED — Will send your version")
    else:
        decision_payload = {"decision": "approve", "reviewed_by": "terminal_user"}
        print(f"\n✅ APPROVED — Will send draft as written")

    # Resume the graph
    print("\n▶ Resuming agent...", end="", flush=True)

    if decision_payload["decision"] == "reject":
        # Re-run — will loop back to draft then review again
        for event in graph.stream(
            Command(resume=decision_payload),
            config=config,
            stream_mode="values"
        ):
            print(".", end="", flush=True)

        # Check for new interrupt (second review round)
        new_state = graph.get_state(config)
        if new_state and new_state.next:
            print("\n\n⏸  NEW DRAFT READY FOR REVIEW")
            new_values = new_state.values
            print(f"\n📝 Revised Draft (v{new_values.get('draft_version', 2)}):")
            print("─" * 50)
            for line in (new_values.get("draft_response", "") or "").split("\n"):
                print(f"  {line}")
            print("─" * 50)

            # Auto-approve the revision for quickstart
            print("\n   [Auto-approving revised draft for quickstart demonstration]")
            for event in graph.stream(
                Command(resume={"decision": "approve", "reviewed_by": "terminal_user"}),
                config=config,
                stream_mode="values"
            ):
                print(".", end="", flush=True)
    else:
        for event in graph.stream(
            Command(resume=decision_payload),
            config=config,
            stream_mode="values"
        ):
            print(".", end="", flush=True)

    print("\n\n✅ COMPLAINT RESOLVED!")

    final_state = graph.get_state(config)
    if final_state and final_state.values:
        v = final_state.values
        print(f"   Final response sent: {'Yes' if v.get('execution_success') else 'No'}")
        print(f"   Sent at           : {v.get('sent_at', 'N/A')[:19]}")
        print(f"   Node path         : {' → '.join(v.get('node_history', []))}")


# ── Run complaints ─────────────────────────────────────────────────────────────
for i, complaint in enumerate(SAMPLE_COMPLAINTS, 1):
    thread_id = f"quickstart_thread_{i}"
    run_complaint_with_terminal_review(complaint, thread_id)
    if i < len(SAMPLE_COMPLAINTS):
        print(f"\n{'─'*60}")
        input("Press Enter to process next complaint...")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("✅ Quick Start Complete!")
print(f"\nWhat you just saw:")
print("  1. Agent classified complaint (category + urgency)")
print("  2. Agent drafted a response using templates")
print("  3. ⏸ Graph PAUSED at REVIEW node (interrupt())")
print("  4. You made a decision (approve/reject/modify)")
print("  5. ▶ Graph RESUMED with your decision")
print("  6. Response was 'sent' and complaint closed")
print(f"\nNext steps:")
print("  • Run the full stack:")
print("      Terminal 1: uvicorn api.main:app --port 8000")
print("      Terminal 2: python demo/app.py")
print("  • Open the Gradio dashboard at http://localhost:7860")
print("  • Submit complaints and review them in the supervisor UI")
print(f"{'='*60}\n")
