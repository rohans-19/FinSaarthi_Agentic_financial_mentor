"""
FinSaarthi — Audit Logger
==========================
Production-grade audit trail for every agent action in the FinSaarthi pipeline.

Design decisions:
    1. SQLite is chosen for zero-setup persistence + portability (single file).
    2. Thread-safe via threading.Lock — safe for concurrent Streamlit sessions.
    3. Context manager support for automatic duration tracking.
    4. Batch export to DataFrame/JSON for hackathon demo & judge review.

File: tools/audit_logger.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_DB_PATH: str = os.getenv("AUDIT_DB_PATH", "./data/finsaarthi_audit.db")

CREATE_TABLE_SQL: str = """
CREATE TABLE IF NOT EXISTS audit_log (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    agent_name      TEXT NOT NULL,
    action          TEXT NOT NULL,
    input_summary   TEXT,
    output_summary  TEXT,
    tools_called    TEXT,
    duration_ms     INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'success',
    error_detail    TEXT,
    metadata_json   TEXT,
    created_at      TEXT NOT NULL
);
"""

CREATE_INDEX_SQL: str = """
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id);
CREATE INDEX IF NOT EXISTS idx_audit_agent   ON audit_log(agent_name);
CREATE INDEX IF NOT EXISTS idx_audit_time    ON audit_log(timestamp);
"""

INSERT_SQL: str = """
INSERT INTO audit_log (
    id, session_id, timestamp, agent_name, action,
    input_summary, output_summary, tools_called,
    duration_ms, status, error_detail, metadata_json, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


class AuditLogger:
    """
    Thread-safe SQLite audit logger for FinSaarthi agent actions.

    Usage — Simple logging:
        logger = AuditLogger(session_id="demo-001")
        logger.log(
            agent_name="portfolio_agent",
            action="parse_cams_pdf",
            input_summary="Uploaded CAMS PDF (2.3 MB)",
            output_summary="Extracted 47 transactions across 5 funds",
            tools_called=["pdfplumber", "pandas"],
            duration_ms=1230,
        )

    Usage — Context-manager with auto-timing:
        logger = AuditLogger(session_id="demo-001")
        with logger.track("tax_agent", "compare_regimes") as tracker:
            result = compute_regime_comparison(...)
            tracker.set_output(f"Old: ₹{result['old']}, New: ₹{result['new']}")
            tracker.set_tools(["scipy", "numpy"])

    Parameters:
        db_path (str): Path to the SQLite database file. Created if missing.
        session_id (str): Unique session identifier. Defaults to a UUID.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        session_id: Optional[str] = None,
    ) -> None:
        self.db_path: str = db_path
        self.session_id: str = session_id or str(uuid.uuid4())
        self._lock: threading.Lock = threading.Lock()

        # Ensure the directory for the database file exists
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the database schema
        self._init_db()

    def _init_db(self) -> None:
        """Create the audit_log table and indexes if they don't exist."""
        with self._get_connection() as conn:
            conn.execute(CREATE_TABLE_SQL)
            conn.executescript(CREATE_INDEX_SQL)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Create a new SQLite connection with optimized pragmas.

        Returns:
            sqlite3.Connection: A configured connection to the audit database.
        """
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def log(
        self,
        agent_name: str,
        action: str,
        input_summary: str = "",
        output_summary: str = "",
        tools_called: Optional[List[str]] = None,
        duration_ms: int = 0,
        status: str = "success",
        error_detail: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Write a single audit entry to the database.

        Parameters:
            agent_name (str): Name of the agent (e.g. "portfolio_agent").
            action (str): Specific action taken (e.g. "parse_cams_pdf").
            input_summary (str): Human-readable summary of inputs.
            output_summary (str): Human-readable summary of outputs.
            tools_called (List[str], optional): List of tools/libraries invoked.
            duration_ms (int): Execution time in milliseconds.
            status (str): "success" | "error" | "warning".
            error_detail (str): Error traceback or message if status is "error".
            metadata (Dict, optional): Additional structured data to store.

        Returns:
            str: The unique ID of the inserted audit entry.
        """
        entry_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        tools_json = json.dumps(tools_called or [])
        metadata_json = json.dumps(metadata or {})

        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    INSERT_SQL,
                    (
                        entry_id,
                        self.session_id,
                        now_iso,
                        agent_name,
                        action,
                        input_summary,
                        output_summary,
                        tools_json,
                        duration_ms,
                        status,
                        error_detail,
                        metadata_json,
                        now_iso,
                    ),
                )
                conn.commit()

        return entry_id

    @contextmanager
    def track(
        self,
        agent_name: str,
        action: str,
        input_summary: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[_ActionTracker, None, None]:
        """
        Context manager that automatically measures execution time and logs
        the result when the block exits.

        Parameters:
            agent_name (str): Name of the agent performing the action.
            action (str): Description of the action being performed.
            input_summary (str): Summary of the input data.
            metadata (Dict, optional): Additional metadata to attach.

        Yields:
            _ActionTracker: A tracker object to set output and tools.

        Example:
            with logger.track("fire_agent", "compute_fire_number") as t:
                result = calculate_fire(...)
                t.set_output(f"FIRE number: ₹{result:,.0f}")
                t.set_tools(["numpy_financial", "scipy"])
        """
        tracker = _ActionTracker()
        start_time = time.perf_counter()

        try:
            yield tracker
        except Exception as exc:
            tracker.status = "error"
            tracker.error_detail = str(exc)
            raise
        finally:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            self.log(
                agent_name=agent_name,
                action=action,
                input_summary=input_summary,
                output_summary=tracker.output_summary,
                tools_called=tracker.tools_called,
                duration_ms=elapsed_ms,
                status=tracker.status,
                error_detail=tracker.error_detail,
                metadata=metadata,
            )

    # ── Query Methods ──────────────────────────────────────────────────────

    def get_session_logs(
        self, session_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all audit entries for a given session.

        Parameters:
            session_id (str, optional): Session to query. Defaults to current.
            limit (int): Maximum number of entries to return.

        Returns:
            List[Dict[str, Any]]: List of audit entries as dictionaries.
        """
        sid = session_id or self.session_id
        query = """
            SELECT id, session_id, timestamp, agent_name, action,
                   input_summary, output_summary, tools_called,
                   duration_ms, status, error_detail, metadata_json
            FROM audit_log
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?;
        """
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (sid, limit)).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def get_agent_logs(
        self, agent_name: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit entries filtered by agent name.

        Parameters:
            agent_name (str): Agent to filter by.
            limit (int): Maximum entries to return.

        Returns:
            List[Dict[str, Any]]: Filtered audit entries.
        """
        query = """
            SELECT id, session_id, timestamp, agent_name, action,
                   input_summary, output_summary, tools_called,
                   duration_ms, status, error_detail, metadata_json
            FROM audit_log
            WHERE agent_name = ?
            ORDER BY timestamp DESC
            LIMIT ?;
        """
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, (agent_name, limit)).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def to_dataframe(
        self, session_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Export audit log as a pandas DataFrame — ideal for Streamlit display
        and hackathon demo dashboards.

        Parameters:
            session_id (str, optional): Filter to this session. If None, returns all.

        Returns:
            pd.DataFrame: Audit log as a DataFrame with parsed JSON columns.
        """
        if session_id:
            query = "SELECT * FROM audit_log WHERE session_id = ? ORDER BY timestamp ASC;"
            params = (session_id,)
        else:
            query = "SELECT * FROM audit_log ORDER BY timestamp ASC;"
            params = ()

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        # Parse JSON columns back to Python objects for display
        if not df.empty:
            df["tools_called"] = df["tools_called"].apply(
                lambda x: json.loads(x) if x else []
            )
            df["metadata_json"] = df["metadata_json"].apply(
                lambda x: json.loads(x) if x else {}
            )

        return df

    def get_session_summary(
        self, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a summary of a session's audit trail — useful for the
        judges' auditability review.

        Parameters:
            session_id (str, optional): Session to summarize. Defaults to current.

        Returns:
            Dict[str, Any]: Summary with total_actions, agents_used,
                            total_duration_ms, error_count, and action_timeline.
        """
        logs = self.get_session_logs(session_id)

        if not logs:
            return {
                "session_id": session_id or self.session_id,
                "total_actions": 0,
                "agents_used": [],
                "total_duration_ms": 0,
                "error_count": 0,
                "action_timeline": [],
            }

        agents_used = list({log["agent_name"] for log in logs})
        total_duration = sum(log.get("duration_ms", 0) for log in logs)
        error_count = sum(1 for log in logs if log.get("status") == "error")
        timeline = [
            {
                "time": log["timestamp"],
                "agent": log["agent_name"],
                "action": log["action"],
                "duration_ms": log["duration_ms"],
                "status": log["status"],
            }
            for log in logs
        ]

        return {
            "session_id": session_id or self.session_id,
            "total_actions": len(logs),
            "agents_used": agents_used,
            "total_duration_ms": total_duration,
            "error_count": error_count,
            "action_timeline": timeline,
        }

    def clear_session(self, session_id: Optional[str] = None) -> int:
        """
        Delete all audit entries for a session. Use with caution.

        Parameters:
            session_id (str, optional): Session to clear. Defaults to current.

        Returns:
            int: Number of rows deleted.
        """
        sid = session_id or self.session_id
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM audit_log WHERE session_id = ?;", (sid,)
                )
                conn.commit()
                return cursor.rowcount

    # ── Internal Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """
        Convert a sqlite3.Row to a dictionary with parsed JSON fields.

        Parameters:
            row (sqlite3.Row): A database row.

        Returns:
            Dict[str, Any]: Dictionary representation with parsed JSON.
        """
        d = dict(row)
        # Parse JSON string fields back to Python objects
        if "tools_called" in d and isinstance(d["tools_called"], str):
            d["tools_called"] = json.loads(d["tools_called"])
        if "metadata_json" in d and isinstance(d["metadata_json"], str):
            d["metadata_json"] = json.loads(d["metadata_json"])
        return d


class _ActionTracker:
    """
    Internal helper used by AuditLogger.track() context manager.
    Allows the caller to set output summary and tools within the `with` block.
    """

    def __init__(self) -> None:
        self.output_summary: str = ""
        self.tools_called: List[str] = []
        self.status: str = "success"
        self.error_detail: str = ""

    def set_output(self, summary: str) -> None:
        """
        Set the output summary for this tracked action.

        Parameters:
            summary (str): Human-readable summary of the action's output.
        """
        self.output_summary = summary

    def set_tools(self, tools: List[str]) -> None:
        """
        Set the list of tools/libraries invoked during this action.

        Parameters:
            tools (List[str]): List of tool names (e.g. ["pdfplumber", "scipy"]).
        """
        self.tools_called = tools

    def set_error(self, error: str) -> None:
        """
        Mark this action as failed and record the error.

        Parameters:
            error (str): Error message or traceback.
        """
        self.status = "error"
        self.error_detail = error
