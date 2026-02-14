"""Ingest settings tab widget."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Button, Static

from lsm.config.models import RootConfig
from lsm.ui.tui.widgets.settings_base import BaseSettingsTab


class IngestSettingsTab(BaseSettingsTab):
    """Settings view for ingest configuration fields."""

    def compose(self) -> ComposeResult:
        yield Container(
            Static("Ingest Settings", classes="settings-section-title"),
            self._field("Persist dir", "settings-ingest-persist-dir"),
            self._field("Collection", "settings-ingest-collection"),
            self._field("Manifest", "settings-ingest-manifest"),
            self._field("Chroma flush interval", "settings-ingest-chroma-flush-interval"),
            self._select_field(
                "Chunking strategy",
                "settings-ingest-chunking-strategy",
                [("structure", "structure"), ("fixed", "fixed")],
            ),
            self._field("Chunk size", "settings-ingest-chunk-size"),
            self._field("Chunk overlap", "settings-ingest-chunk-overlap"),
            self._field("Tags per chunk", "settings-ingest-tags-per-chunk"),
            self._field("Translation target", "settings-ingest-translation-target"),
            self._field("Extensions (comma-separated)", "settings-ingest-extensions"),
            self._field("Exclude dirs (comma-separated)", "settings-ingest-exclude-dirs"),
            self._field("Max files", "settings-ingest-max-files"),
            self._field("Max seconds", "settings-ingest-max-seconds"),
            self._field("Override extensions", "settings-ingest-override-extensions", field_type="switch"),
            self._field("Override excludes", "settings-ingest-override-excludes", field_type="switch"),
            self._field("Dry run", "settings-ingest-dry-run", field_type="switch"),
            self._field("Skip errors", "settings-ingest-skip-errors", field_type="switch"),
            self._field("Enable OCR", "settings-ingest-enable-ocr", field_type="switch"),
            self._field("Enable AI tagging", "settings-ingest-enable-ai-tagging", field_type="switch"),
            self._field(
                "Enable language detection",
                "settings-ingest-enable-language-detection",
                field_type="switch",
            ),
            self._field("Enable translation", "settings-ingest-enable-translation", field_type="switch"),
            self._field("Enable versioning", "settings-ingest-enable-versioning", field_type="switch"),
            Static("Roots", classes="settings-subsection-title"),
            Vertical(id="settings-ingest-roots-list"),
            Horizontal(Button("Add Root", id="settings-ingest-root-add"), classes="settings-actions"),
            self._save_reset_row("ingest"),
            classes="settings-section",
        )

    def refresh_fields(self, config: Any) -> None:
        ingest = getattr(config, "ingest", None)
        if ingest is None:
            return

        self._refresh_roots_fields(config)
        self._set_input("settings-ingest-persist-dir", str(getattr(ingest, "persist_dir", "")))
        self._set_input("settings-ingest-collection", str(getattr(ingest, "collection", "")))
        self._set_input("settings-ingest-manifest", str(getattr(ingest, "manifest", "")))
        self._set_input(
            "settings-ingest-chroma-flush-interval",
            str(getattr(ingest, "chroma_flush_interval", "")),
        )
        self._set_select_value(
            "settings-ingest-chunking-strategy",
            str(getattr(ingest, "chunking_strategy", "")),
        )
        self._set_input("settings-ingest-chunk-size", str(getattr(ingest, "chunk_size", "")))
        self._set_input("settings-ingest-chunk-overlap", str(getattr(ingest, "chunk_overlap", "")))
        self._set_input("settings-ingest-tags-per-chunk", str(getattr(ingest, "tags_per_chunk", "")))
        self._set_input(
            "settings-ingest-translation-target",
            str(getattr(ingest, "translation_target", "")),
        )
        self._set_input("settings-ingest-extensions", ", ".join(getattr(ingest, "extensions", None) or []))
        self._set_input(
            "settings-ingest-exclude-dirs",
            ", ".join(getattr(ingest, "exclude_dirs", None) or []),
        )
        self._set_input(
            "settings-ingest-max-files",
            self._format_optional(getattr(ingest, "max_files", None)),
        )
        self._set_input(
            "settings-ingest-max-seconds",
            self._format_optional(getattr(ingest, "max_seconds", None)),
        )
        self._set_switch(
            "settings-ingest-override-extensions",
            bool(getattr(ingest, "override_extensions", False)),
        )
        self._set_switch(
            "settings-ingest-override-excludes",
            bool(getattr(ingest, "override_excludes", False)),
        )
        self._set_switch("settings-ingest-dry-run", bool(getattr(ingest, "dry_run", False)))
        self._set_switch("settings-ingest-skip-errors", bool(getattr(ingest, "skip_errors", True)))
        self._set_switch("settings-ingest-enable-ocr", bool(getattr(ingest, "enable_ocr", False)))
        self._set_switch(
            "settings-ingest-enable-ai-tagging",
            bool(getattr(ingest, "enable_ai_tagging", False)),
        )
        self._set_switch(
            "settings-ingest-enable-language-detection",
            bool(getattr(ingest, "enable_language_detection", False)),
        )
        self._set_switch(
            "settings-ingest-enable-translation",
            bool(getattr(ingest, "enable_translation", False)),
        )
        self._set_switch(
            "settings-ingest-enable-versioning",
            bool(getattr(ingest, "enable_versioning", False)),
        )

    def apply_update(self, field_id: str, value: Any, config: Any) -> bool:
        ingest = getattr(config, "ingest", None)
        if ingest is None:
            return False

        text = str(value or "").strip()

        root_match = re.fullmatch(r"settings-ingest-root-(\d+)-([a-z-]+)", field_id)
        if root_match:
            idx = int(root_match.group(1))
            field = root_match.group(2)
            roots = ingest.roots or []
            if idx >= len(roots):
                return True
            root = roots[idx]
            if field == "path":
                if not text:
                    raise ValueError("root path cannot be empty")
                root.path = Path(text)
            elif field == "tags":
                root.tags = self._parse_csv(text)
            elif field == "content-type":
                root.content_type = text or None
            return True

        if field_id == "settings-ingest-persist-dir":
            ingest.persist_dir = Path(text)
            return True
        if field_id == "settings-ingest-collection":
            ingest.collection = text
            return True
        if field_id == "settings-ingest-manifest":
            ingest.manifest = Path(text)
            return True
        if field_id == "settings-ingest-chroma-flush-interval" and text:
            ingest.chroma_flush_interval = int(text)
            return True
        if field_id == "settings-ingest-chunking-strategy":
            ingest.chunking_strategy = text
            return True
        if field_id == "settings-ingest-chunk-size" and text:
            ingest.chunk_size = int(text)
            return True
        if field_id == "settings-ingest-chunk-overlap" and text:
            ingest.chunk_overlap = int(text)
            return True
        if field_id == "settings-ingest-tags-per-chunk" and text:
            ingest.tags_per_chunk = int(text)
            return True
        if field_id == "settings-ingest-translation-target":
            ingest.translation_target = text or "en"
            return True
        if field_id == "settings-ingest-extensions":
            ingest.extensions = self._parse_csv(text)
            return True
        if field_id == "settings-ingest-exclude-dirs":
            ingest.exclude_dirs = self._parse_csv(text)
            return True
        if field_id == "settings-ingest-max-files":
            ingest.max_files = int(text) if text else None
            return True
        if field_id == "settings-ingest-max-seconds":
            ingest.max_seconds = int(text) if text else None
            return True
        if field_id == "settings-ingest-override-extensions":
            ingest.override_extensions = bool(value)
            return True
        if field_id == "settings-ingest-override-excludes":
            ingest.override_excludes = bool(value)
            return True
        if field_id == "settings-ingest-dry-run":
            ingest.dry_run = bool(value)
            return True
        if field_id == "settings-ingest-skip-errors":
            ingest.skip_errors = bool(value)
            return True
        if field_id == "settings-ingest-enable-ocr":
            ingest.enable_ocr = bool(value)
            return True
        if field_id == "settings-ingest-enable-ai-tagging":
            ingest.enable_ai_tagging = bool(value)
            return True
        if field_id == "settings-ingest-enable-language-detection":
            ingest.enable_language_detection = bool(value)
            return True
        if field_id == "settings-ingest-enable-translation":
            ingest.enable_translation = bool(value)
            return True
        if field_id == "settings-ingest-enable-versioning":
            ingest.enable_versioning = bool(value)
            return True
        return False

    def handle_button(self, button_id: str, config: Any) -> bool:
        if button_id == "settings-ingest-root-add":
            self._add_root(config)
            return True
        root_remove = re.fullmatch(r"settings-ingest-root-remove-(\d+)", button_id)
        if root_remove:
            self._remove_root(config, int(root_remove.group(1)))
            return True
        return False

    def _refresh_roots_fields(self, config: Any) -> None:
        roots = list(getattr(config.ingest, "roots", None) or [])
        root_widgets = []
        if not roots:
            root_widgets.append(Static("No ingest roots configured.", classes="settings-label"))

        for idx, root in enumerate(roots):
            tags = ", ".join(getattr(root, "tags", None) or [])
            root_widgets.append(
                Container(
                    Static(f"Root {idx + 1}", classes="settings-subsection-title"),
                    self._field(
                        "Path",
                        f"settings-ingest-root-{idx}-path",
                        value=str(getattr(root, "path", "")),
                    ),
                    self._field(
                        "Tags (comma-separated)",
                        f"settings-ingest-root-{idx}-tags",
                        value=tags,
                    ),
                    self._field(
                        "Content type",
                        f"settings-ingest-root-{idx}-content-type",
                        value=self._format_optional(getattr(root, "content_type", None)),
                    ),
                    Horizontal(
                        Button("Remove Root", id=f"settings-ingest-root-remove-{idx}"),
                        classes="settings-actions",
                    ),
                    classes="settings-subsection",
                )
            )

        self._replace_container_children("#settings-ingest-roots-list", root_widgets)

    def _add_root(self, config: Any) -> None:
        config.ingest.roots.append(RootConfig(path=Path(".")))
        self._refresh_roots_fields(config)
        self.post_status("Added ingest root", False)

    def _remove_root(self, config: Any, idx: int) -> None:
        roots = config.ingest.roots or []
        if idx >= len(roots):
            return
        if len(roots) <= 1:
            raise ValueError("At least one ingest root is required.")
        roots.pop(idx)
        self._refresh_roots_fields(config)
        self.post_status("Removed ingest root", False)

    @staticmethod
    def _parse_csv(text: str) -> Optional[list[str]]:
        values = [item.strip() for item in text.split(",") if item.strip()]
        return values or None

    @staticmethod
    def _format_optional(value: Optional[Any]) -> str:
        return "" if value is None else str(value)
