"""Shared base screen mixin for lifecycle, workers, timers, and status helpers.

Provides standardized hooks so concrete screen widgets can register/cancel
workers and timers through the app's managed lifecycle API without duplicating
the delegation boilerplate.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from textual.timer import Timer
from textual.widget import Widget

from lsm.logging import get_logger

logger = get_logger(__name__)


class ManagedScreenMixin:
    """Mixin providing worker/timer lifecycle delegation for screen widgets.

    Concrete screens should inherit from both ``Widget`` and this mixin:

        class MyScreen(ManagedScreenMixin, Widget):
            ...

    The mixin assumes the host object has ``self.app``, ``self.run_worker``,
    and ``self.set_interval`` (provided by ``Widget``).
    """

    # ------------------------------------------------------------------
    # Owner token
    # ------------------------------------------------------------------

    def _worker_owner_token(self) -> str:
        """Return a stable owner identifier for this screen."""
        widget_id = str(getattr(self, "id", "") or "").strip()
        if widget_id:
            return widget_id
        return self.__class__.__name__

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def _start_managed_worker(
        self,
        *,
        worker_key: str,
        work_factory: Callable[[], Any],
        timeout_s: float,
        exclusive: bool,
    ) -> None:
        """Start a background worker registered with the app lifecycle.

        Falls back to an unmanaged ``run_worker`` when the app does not
        expose the managed API.

        Args:
            worker_key: Key for this worker (unique per owner).
            work_factory: Callable that returns a coroutine to run.
            timeout_s: Timeout for join on cancellation.
            exclusive: Whether the worker is exclusive.
        """
        app_obj = getattr(self, "app", None)
        starter = getattr(app_obj, "start_managed_worker", None)
        run_worker_fn = getattr(self, "run_worker", None)
        if callable(starter) and callable(run_worker_fn):
            try:
                starter(
                    owner=self._worker_owner_token(),
                    key=worker_key,
                    timeout_s=timeout_s,
                    start=lambda: run_worker_fn(work_factory(), exclusive=exclusive),
                )
                return
            except Exception:
                logger.exception(
                    "Failed to start managed worker '%s' for %s.",
                    worker_key,
                    self._worker_owner_token(),
                )
        if callable(run_worker_fn):
            run_worker_fn(work_factory(), exclusive=exclusive)

    def _register_managed_worker(
        self,
        *,
        key: str,
        worker: Any,
        timeout_s: float,
    ) -> None:
        """Register an already-started worker with the app lifecycle.

        Args:
            key: Worker key.
            worker: Worker object.
            timeout_s: Timeout for join on cancellation.
        """
        app_obj = getattr(self, "app", None)
        register = getattr(app_obj, "register_managed_worker", None)
        if not callable(register):
            return
        try:
            register(
                owner=self._worker_owner_token(),
                key=key,
                worker=worker,
                timeout_s=timeout_s,
            )
        except Exception:
            logger.exception(
                "Failed to register managed worker '%s' for %s.",
                key,
                self._worker_owner_token(),
            )

    def _clear_managed_worker(self, key: str) -> None:
        """Remove a worker from the registry without cancelling.

        Args:
            key: Worker key to clear.
        """
        app_obj = getattr(self, "app", None)
        clear = getattr(app_obj, "clear_managed_worker", None)
        if not callable(clear):
            return
        try:
            clear(owner=self._worker_owner_token(), key=key)
        except Exception:
            logger.exception(
                "Failed to clear managed worker '%s' for %s.",
                key,
                self._worker_owner_token(),
            )

    def _cancel_managed_workers(self, *, reason: str) -> None:
        """Cancel all workers owned by this screen.

        Args:
            reason: Human-readable reason for cancellation.
        """
        app_obj = getattr(self, "app", None)
        cancel_owner = getattr(app_obj, "cancel_managed_workers_for_owner", None)
        if not callable(cancel_owner):
            return
        try:
            results = cancel_owner(
                owner=self._worker_owner_token(),
                reason=reason,
            )
        except Exception:
            logger.exception(
                "Failed to cancel managed workers for %s (%s).",
                self._worker_owner_token(),
                reason,
            )
            return
        if any(not bool(result) for result in results.values()):
            logger.warning(
                "Worker shutdown hit timeout for owner '%s'.",
                self._worker_owner_token(),
            )

    # ------------------------------------------------------------------
    # Timers
    # ------------------------------------------------------------------

    def _start_timer(
        self,
        *,
        interval_seconds: float,
        callback: Any,
    ) -> Optional[Timer]:
        """Start an unmanaged periodic timer.

        Args:
            interval_seconds: Interval between callbacks.
            callback: Timer callback.

        Returns:
            Timer object or None on failure.
        """
        set_interval_fn = getattr(self, "set_interval", None)
        if not callable(set_interval_fn):
            return None
        try:
            return set_interval_fn(interval_seconds, callback)
        except Exception:
            return None

    def _start_managed_timer(
        self,
        *,
        key: str,
        interval_seconds: float,
        callback: Any,
    ) -> Optional[Timer]:
        """Start a periodic timer registered with the app lifecycle.

        Falls back to an unmanaged timer when the app does not expose the
        managed API.

        Args:
            key: Timer key (unique per owner).
            interval_seconds: Interval between callbacks.
            callback: Timer callback.

        Returns:
            Timer object or None on failure.
        """
        app_obj = getattr(self, "app", None)
        start_timer = getattr(app_obj, "start_managed_timer", None)
        if callable(start_timer):
            try:
                return start_timer(
                    owner=self._worker_owner_token(),
                    key=key,
                    start=lambda: self._start_timer(
                        interval_seconds=interval_seconds,
                        callback=callback,
                    ),
                    restart=False,
                )
            except Exception:
                logger.exception("Failed to start managed timer '%s'.", key)
        return self._start_timer(
            interval_seconds=interval_seconds,
            callback=callback,
        )

    def _stop_timer(self, timer: Optional[Timer]) -> None:
        """Stop a single timer.

        Args:
            timer: Timer object to stop, or None.
        """
        if timer is None:
            return
        try:
            timer.stop()
        except Exception:
            return

    def _stop_managed_timer(
        self,
        *,
        key: str,
        timer: Optional[Timer],
        reason: str,
    ) -> None:
        """Stop a specific managed timer by key.

        Falls back to direct timer stop when the app does not expose the
        managed API.

        Args:
            key: Timer key.
            timer: Timer object to stop as fallback.
            reason: Human-readable reason.
        """
        app_obj = getattr(self, "app", None)
        stop_timer = getattr(app_obj, "stop_managed_timer", None)
        if callable(stop_timer):
            try:
                stop_timer(
                    owner=self._worker_owner_token(),
                    key=key,
                    reason=reason,
                )
                return
            except Exception:
                logger.exception("Failed to stop managed timer '%s'.", key)
        self._stop_timer(timer)

    def _cancel_managed_timers(self, *, reason: str) -> None:
        """Stop all timers owned by this screen.

        Args:
            reason: Human-readable reason for stopping.
        """
        app_obj = getattr(self, "app", None)
        stop_owner = getattr(app_obj, "stop_managed_timers_for_owner", None)
        if not callable(stop_owner):
            return
        try:
            stop_owner(
                owner=self._worker_owner_token(),
                reason=reason,
            )
        except Exception:
            logger.exception(
                "Failed to stop managed timers for %s (%s).",
                self._worker_owner_token(),
                reason,
            )

    def _stop_managed_timers(self, *, reason: str) -> None:
        """Alias for ``_cancel_managed_timers`` used by agents screen."""
        self._cancel_managed_timers(reason=reason)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _update_static(self, selector: str, message: str) -> None:
        """Update a Static widget by selector.

        Args:
            selector: CSS selector for the Static widget.
            message: New message text.
        """
        try:
            from textual.widgets import Static
            widget = getattr(self, "query_one", lambda *a, **k: None)(selector, Static)
            if widget is not None:
                widget.update(message)
        except Exception:
            pass
