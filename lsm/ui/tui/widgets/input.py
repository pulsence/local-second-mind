"""
Command input widget for LSM TUI.

Provides input with:
- Command history (up/down navigation)
- Tab completion popup
- Command validation
"""

from __future__ import annotations

from typing import List, Optional, Callable

from textual.app import ComposeResult
from textual.widgets import Input
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding

from lsm.logging import get_logger

logger = get_logger(__name__)


class CommandSubmitted(Message):
    """Message sent when a command is submitted."""

    def __init__(self, command: str) -> None:
        self.command = command
        super().__init__()


class CommandInput(Widget):
    """
    Command input widget with history and autocomplete.

    Features:
    - Up/Down arrow for command history
    - Tab completion for commands
    - Escape to clear
    """

    DEFAULT_CSS = """
    CommandInput {
        height: auto;
        min-height: 3;
        max-height: 10;
        background: $surface-darken-1;
        border: tall $primary-darken-1;
        padding: 0 1;
    }

    CommandInput:focus-within {
        border: tall $primary;
    }
    """

    BINDINGS = [
        Binding("up", "history_prev", "Previous", show=False),
        Binding("down", "history_next", "Next", show=False),
        Binding("tab", "autocomplete", "Complete", show=False),
        Binding("escape", "clear", "Clear", show=False),
    ]

    # Reactive state
    history_index: reactive[int] = reactive(-1)

    def __init__(
        self,
        placeholder: str = "Enter command...",
        completer: Optional[Callable[[str], List[str]]] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize the command input.

        Args:
            placeholder: Placeholder text
            completer: Optional autocomplete function
        """
        super().__init__(*args, **kwargs)
        self._placeholder = placeholder
        self._completer = completer
        self._history: List[str] = []
        self._history_max = 100
        self._current_input = ""

    def compose(self) -> ComposeResult:
        """Compose the input widget."""
        yield Input(
            placeholder=self._placeholder,
            id="command-input-field",
        )

    @property
    def value(self) -> str:
        """Get the current input value."""
        return self.query_one("#command-input-field", Input).value

    @value.setter
    def value(self, val: str) -> None:
        """Set the input value."""
        self.query_one("#command-input-field", Input).value = val

    def focus(self) -> None:
        """Focus the input field."""
        self.query_one("#command-input-field", Input).focus()

    def clear(self) -> None:
        """Clear the input field."""
        self.value = ""
        self.history_index = -1

    def add_to_history(self, command: str) -> None:
        """
        Add a command to history.

        Args:
            command: Command to add
        """
        # Don't add empty or duplicate commands
        if not command.strip():
            return
        if self._history and self._history[-1] == command:
            return

        self._history.append(command)

        # Trim history if needed
        if len(self._history) > self._history_max:
            self._history = self._history[-self._history_max:]

        # Reset history index
        self.history_index = -1

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        command = event.value.strip()
        if command:
            self.add_to_history(command)
            self.post_message(CommandSubmitted(command))
        event.input.value = ""

    def action_history_prev(self) -> None:
        """Navigate to previous history item."""
        if not self._history:
            return

        # Save current input if starting history navigation
        if self.history_index == -1:
            self._current_input = self.value

        # Move back in history
        if self.history_index < len(self._history) - 1:
            self.history_index += 1
            self.value = self._history[-(self.history_index + 1)]

    def action_history_next(self) -> None:
        """Navigate to next history item."""
        if self.history_index > 0:
            self.history_index -= 1
            self.value = self._history[-(self.history_index + 1)]
        elif self.history_index == 0:
            self.history_index = -1
            self.value = self._current_input

    def action_autocomplete(self) -> None:
        """Trigger autocomplete."""
        if not self._completer:
            return

        current = self.value
        completions = self._completer(current)

        if not completions:
            return

        if len(completions) == 1:
            # Single completion - apply it
            self.value = completions[0]
        else:
            # Multiple completions - find common prefix
            common = self._common_prefix(completions)
            if len(common) > len(current):
                self.value = common
            else:
                # Show completion options (via notification for now)
                self.app.notify(
                    f"Completions: {', '.join(completions[:5])}{'...' if len(completions) > 5 else ''}",
                    timeout=3,
                )

    def action_clear(self) -> None:
        """Clear the input."""
        self.clear()

    @staticmethod
    def _common_prefix(strings: List[str]) -> str:
        """
        Find the common prefix of a list of strings.

        Args:
            strings: List of strings

        Returns:
            Common prefix
        """
        if not strings:
            return ""
        if len(strings) == 1:
            return strings[0]

        prefix = strings[0]
        for s in strings[1:]:
            while not s.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix
