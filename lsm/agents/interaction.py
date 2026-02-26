"""
Thread-safe interaction channel between agent runtime threads and UI threads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Optional

_VALID_REQUEST_TYPES = {"permission", "clarification", "feedback", "confirmation"}
_VALID_DECISIONS = {"approve", "deny", "approve_session", "reply"}
_VALID_TIMEOUT_ACTIONS = {"deny", "approve"}


@dataclass(frozen=True)
class InteractionRequest:
    """
    Request posted by the harness to the UI.
    """

    request_id: str
    request_type: str
    tool_name: Optional[str] = None
    risk_level: Optional[str] = None
    reason: Optional[str] = None
    args_summary: Optional[str] = None
    prompt: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        request_id = str(self.request_id or "").strip()
        if not request_id:
            raise ValueError("InteractionRequest.request_id is required")
        request_type = str(self.request_type or "").strip().lower()
        if request_type not in _VALID_REQUEST_TYPES:
            raise ValueError(
                "InteractionRequest.request_type must be one of: "
                "permission, clarification, feedback, confirmation"
            )
        tool_name = self._normalize_optional(self.tool_name)
        risk_level = self._normalize_optional(self.risk_level)
        reason = self._normalize_optional(self.reason)
        args_summary = self._normalize_optional(self.args_summary)
        prompt = str(self.prompt or "").strip()
        timestamp = self.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "request_type", request_type)
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "risk_level", risk_level)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "args_summary", args_summary)
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "timestamp", timestamp)

    @staticmethod
    def _normalize_optional(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None


@dataclass(frozen=True)
class InteractionResponse:
    """
    Response posted by the UI back to the harness.
    """

    request_id: str
    decision: str
    user_message: str = ""

    def __post_init__(self) -> None:
        request_id = str(self.request_id or "").strip()
        if not request_id:
            raise ValueError("InteractionResponse.request_id is required")
        decision = str(self.decision or "").strip().lower()
        if decision not in _VALID_DECISIONS:
            raise ValueError(
                "InteractionResponse.decision must be one of: "
                "approve, deny, approve_session, reply"
            )
        user_message = str(self.user_message or "").strip()
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "user_message", user_message)


class InteractionChannel:
    """
    In-memory interaction bridge for a single agent run.

    Harness thread behavior:
    - Call `post_request(...)` and block until UI posts a response.

    UI thread behavior:
    - Poll with `has_pending()` / `get_pending_request()`.
    - Reply with `post_response(...)`.
    """

    def __init__(
        self,
        timeout_seconds: float = 300,
        timeout_action: str = "deny",
        acknowledged_timeout_seconds: float = 0,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.timeout_action = str(timeout_action or "deny").strip().lower()
        self.acknowledged_timeout_seconds = float(acknowledged_timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.timeout_action not in _VALID_TIMEOUT_ACTIONS:
            raise ValueError("timeout_action must be one of: deny, approve")
        if self.acknowledged_timeout_seconds < 0:
            raise ValueError("acknowledged_timeout_seconds must be >= 0")

        self._lock = threading.Lock()
        self._pending_request: Optional[InteractionRequest] = None
        self._pending_response: Optional[InteractionResponse] = None
        self._pending_event: Optional[threading.Event] = None
        self._session_approvals: set[str] = set()
        self._shutdown = False
        self._shutdown_reason = ""
        self._acknowledged = False
        self._acknowledged_at: Optional[datetime] = None

    def post_request(self, request: InteractionRequest) -> InteractionResponse:
        """
        Post a request and block until a response arrives or timeout expires.

        Uses a two-phase timeout:
        - Phase 1: Wait up to timeout_seconds for acknowledgment or response
        - Phase 2: Once acknowledged, wait up to acknowledged_timeout_seconds (0 = infinite)

        Args:
            request: Request to send to UI.

        Returns:
            InteractionResponse chosen by the user (or synthesized on timeout).

        Raises:
            PermissionError: If timeout action is deny.
            RuntimeError: If the channel is shut down or request state is invalid.
        """
        import time

        POLL_INTERVAL = 0.5

        with self._lock:
            if self._shutdown:
                reason = self._shutdown_reason or "interaction channel is shut down"
                raise RuntimeError(reason)

            if (
                request.request_type == "permission"
                and request.tool_name is not None
                and request.tool_name in self._session_approvals
            ):
                return InteractionResponse(
                    request_id=request.request_id,
                    decision="approve_session",
                    user_message="Previously approved for this session.",
                )

            if self._pending_request is not None:
                raise RuntimeError("Cannot post a new interaction request while another is pending")

            event = threading.Event()
            self._pending_request = request
            self._pending_response = None
            self._pending_event = event
            self._acknowledged = False
            self._acknowledged_at = None

        start_time = time.monotonic()

        while True:
            if event.wait(timeout=POLL_INTERVAL):
                response = self._consume_response(event)
                if response is None:
                    raise RuntimeError("Interaction channel signaled without a response")
                return response

            elapsed = time.monotonic() - start_time

            with self._lock:
                acknowledged = self._acknowledged
                acknowledged_at = self._acknowledged_at
                if self._pending_event is not event:
                    response = self._pending_response
                    if response is not None:
                        return response

            if not acknowledged:
                if elapsed >= self.timeout_seconds:
                    with self._lock:
                        if self._pending_event is event:
                            self._clear_pending_locked()

                    if self.timeout_action == "approve":
                        return InteractionResponse(
                            request_id=request.request_id,
                            decision="approve",
                            user_message="Auto-approved because interaction request timed out.",
                        )
                    raise PermissionError(
                        f"Interaction request '{request.request_id}' timed out after "
                        f"{self.timeout_seconds} seconds"
                    )
            else:
                if self.acknowledged_timeout_seconds > 0 and acknowledged_at is not None:
                    acknowledged_elapsed = (datetime.now(timezone.utc) - acknowledged_at).total_seconds()
                    if acknowledged_elapsed >= self.acknowledged_timeout_seconds:
                        with self._lock:
                            if self._pending_event is event:
                                self._clear_pending_locked()

                        if self.timeout_action == "approve":
                            return InteractionResponse(
                                request_id=request.request_id,
                                decision="approve",
                                user_message="Auto-approved because acknowledged interaction timed out.",
                            )
                        raise PermissionError(
                            f"Acknowledged interaction request '{request.request_id}' timed out after "
                            f"{self.acknowledged_timeout_seconds} seconds"
                        )

    def acknowledge_request(self, request_id: str) -> None:
        """
        Mark the current pending request as acknowledged.

        Once acknowledged, the request uses acknowledged_timeout_seconds instead of
        timeout_seconds. If acknowledged_timeout_seconds is 0, the request waits
        indefinitely.

        Args:
            request_id: The ID of the request to acknowledge.

        Note:
            If request_id does not match the pending request, this is a no-op.
        """
        with self._lock:
            if self._pending_request is not None and self._pending_request.request_id == request_id:
                self._acknowledged = True
                self._acknowledged_at = datetime.now(timezone.utc)

    def get_pending_request(self) -> Optional[InteractionRequest]:
        """
        Return the current pending request, if any.
        """
        with self._lock:
            return self._pending_request

    def post_response(self, response: InteractionResponse) -> None:
        """
        Post a UI response for the current pending request.
        """
        with self._lock:
            pending = self._pending_request
            event = self._pending_event
            if pending is None or event is None:
                raise ValueError("No pending interaction request to respond to")
            if response.request_id != pending.request_id:
                raise ValueError(
                    "Interaction response request_id does not match pending request"
                )
            self._pending_response = response
            if response.decision == "approve_session" and pending.tool_name is not None:
                self._session_approvals.add(pending.tool_name)
            event.set()

    def has_pending(self) -> bool:
        """
        Return True when a request is waiting for UI action.
        """
        with self._lock:
            return self._pending_request is not None

    def cancel_pending(self, reason: str) -> bool:
        """
        Cancel the current pending request and unblock waiters.

        Returns:
            True when a request was cancelled, False when nothing was pending.
        """
        with self._lock:
            pending = self._pending_request
            event = self._pending_event
            if pending is None or event is None:
                return False
            self._pending_response = InteractionResponse(
                request_id=pending.request_id,
                decision="deny",
                user_message=str(reason or "").strip() or "Interaction request cancelled.",
            )
            event.set()
            return True

    def shutdown(self, reason: str) -> None:
        """
        Mark channel closed and cancel any pending request.
        """
        with self._lock:
            self._shutdown = True
            self._shutdown_reason = str(reason or "").strip() or "Interaction channel shut down."
        self.cancel_pending(self._shutdown_reason)

    def has_session_approval(self, tool_name: str) -> bool:
        """
        Check whether a tool has been approved for this session.
        """
        normalized = str(tool_name or "").strip()
        if not normalized:
            return False
        with self._lock:
            return normalized in self._session_approvals

    def approve_for_session(self, tool_name: str) -> None:
        """
        Record session-wide approval for a tool.
        """
        normalized = str(tool_name or "").strip()
        if not normalized:
            raise ValueError("tool_name is required")
        with self._lock:
            self._session_approvals.add(normalized)

    def _consume_response(self, event: threading.Event) -> Optional[InteractionResponse]:
        with self._lock:
            if self._pending_event is not event:
                return self._pending_response
            response = self._pending_response
            self._clear_pending_locked()
            return response

    def _clear_pending_locked(self) -> None:
        self._pending_request = None
        self._pending_response = None
        self._pending_event = None
