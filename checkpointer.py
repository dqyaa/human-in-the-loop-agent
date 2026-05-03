"""
Checkpointer & Thread Manager
================================
Manages SQLite persistence for LangGraph.

The checkpointer is the backbone of human-in-the-loop:
  - Saves the COMPLETE graph state after every node
  - Allows pausing for hours/days and resuming exactly where you left off
  - Enables audit trails (every state transition is logged)
  - Provides TTL-based thread expiry (auto-reject stale approvals)

Think of it like a video game save system:
  - Before REVIEW node: "checkpoint saved"
  - Human goes offline for 2 days
  - Human comes back, loads checkpoint, resumes

Usage:
    from agent.checkpointer import get_checkpointer, ThreadManager

    checkpointer = get_checkpointer("complaints.db")
    graph = build_graph(checkpointer=checkpointer)

    manager = ThreadManager(checkpointer)
    pending = manager.get_pending_reviews()
"""

import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def get_checkpointer(db_path: str = "complaints.db"):
    """
    Get a SQLite checkpointer for LangGraph.

    SQLiteSaver persists ALL graph state to a SQLite database.
    This means:
    - Server restarts don't lose pending approvals
    - You can query pending reviews from any process
    - Full audit trail of every state transition
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        logger.info(f"SQLite checkpointer ready: {db_path}")
        return checkpointer
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-sqlite not installed. "
            "Using in-memory checkpointer (state lost on restart).\n"
            "Install: pip install langgraph-checkpoint-sqlite"
        )
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


class ThreadManager:
    """
    Manages complaint workflow threads.

    A thread = one complete complaint handling workflow instance.
    Each complaint gets a unique thread_id.
    The checkpointer stores state per thread_id.

    Methods:
        create_thread()     — Start a new complaint workflow
        get_pending()       — List complaints waiting for human review
        get_thread_state()  — Get current state of a thread
        expire_old()        — Auto-reject threads older than TTL
    """

    def __init__(self, checkpointer, db_path: str = "complaints.db"):
        self.checkpointer = checkpointer
        self.db_path = db_path
        self._init_meta_db()

    def _init_meta_db(self):
        """Create metadata table for thread tracking."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS complaint_threads (
                        thread_id    TEXT PRIMARY KEY,
                        complaint_id TEXT,
                        customer     TEXT,
                        category     TEXT,
                        urgency      TEXT,
                        status       TEXT DEFAULT 'pending_review',
                        created_at   TEXT,
                        updated_at   TEXT,
                        resolved_at  TEXT,
                        metadata     TEXT
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_status ON complaint_threads(status)"
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"Meta DB init: {e}")

    def register_thread(
        self,
        thread_id: str,
        complaint_id: str,
        customer: str,
        category: str,
        urgency: str,
    ):
        """Register a new thread in the metadata table."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO complaint_threads
                    (thread_id, complaint_id, customer, category, urgency,
                     status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 'pending_review', ?, ?)
                """, (thread_id, complaint_id, customer, category, urgency, now, now))
                conn.commit()
        except Exception as e:
            logger.error(f"Thread registration error: {e}")

    def update_thread_status(self, thread_id: str, status: str):
        """Update thread status."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE complaint_threads
                    SET status = ?, updated_at = ?
                    WHERE thread_id = ?
                """, (status, now, thread_id))
                if status == "resolved":
                    conn.execute(
                        "UPDATE complaint_threads SET resolved_at = ? WHERE thread_id = ?",
                        (now, thread_id)
                    )
                conn.commit()
        except Exception as e:
            logger.error(f"Thread status update error: {e}")

    def get_pending_reviews(self) -> list[dict]:
        """Get all threads currently waiting for human approval."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT * FROM complaint_threads
                    WHERE status = 'pending_review'
                    ORDER BY
                        CASE urgency
                            WHEN 'critical' THEN 1
                            WHEN 'high'     THEN 2
                            WHEN 'medium'   THEN 3
                            ELSE            4
                        END,
                        created_at ASC
                """).fetchall()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"Get pending error: {e}")
            return []

    def expire_old_threads(self, ttl_hours: int = 24) -> int:
        """
        Auto-reject threads not reviewed within TTL.

        Production best practice: threads waiting for human approval
        should not wait indefinitely. After TTL, auto-reject and
        notify the original requester.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
        ).isoformat()

        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("""
                    UPDATE complaint_threads
                    SET status = 'expired', updated_at = ?
                    WHERE status = 'pending_review' AND created_at < ?
                """, (datetime.now(timezone.utc).isoformat(), cutoff))
                count = result.rowcount
                conn.commit()
            if count > 0:
                logger.warning(f"Expired {count} thread(s) older than {ttl_hours}h")
            return count
        except Exception as e:
            logger.error(f"Expire error: {e}")
            return 0

    def get_stats(self) -> dict:
        """Get dashboard statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT
                        status,
                        COUNT(*) as count,
                        AVG(CASE urgency WHEN 'high' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END) as avg_urgency
                    FROM complaint_threads
                    GROUP BY status
                """).fetchall()
                return {r["status"]: r["count"] for r in rows}
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {}
