from __future__ import annotations

from lsm.ingest.progress import Progress


def test_start_and_finish_emit_events() -> None:
    events: list[tuple[str, int, int, str]] = []
    progress = Progress(on_update=lambda e, c, t, m: events.append((e, c, t, m)))

    progress.start(3)
    progress.finish()

    assert events[0][0] == "start"
    assert events[0][2] == 3
    assert events[-1][0] == "finish"


def test_file_counters_and_forced_progress_emit() -> None:
    events: list[tuple[str, int, int, str]] = []
    progress = Progress(report_every_s=999.0, on_update=lambda e, c, t, m: events.append((e, c, t, m)))
    progress.start(4)

    progress.file_updated(5)
    progress.file_skipped()
    progress.file_empty()
    progress.error("boom")
    progress.write_done(5)
    progress.maybe_report(force=True)

    assert progress.seen == 3
    assert progress.updated == 1
    assert progress.skipped == 1
    assert progress.empty == 1
    assert progress.errors == 1
    assert progress.chunks == 5
    assert progress.writes == 1
    assert any(e[0] == "error" and "boom" in e[3] for e in events)
    assert any(e[0] == "progress" and "updated=1 skipped=1 empty=1 errors=1" in e[3] for e in events)


def test_maybe_report_respects_interval() -> None:
    events: list[tuple[str, int, int, str]] = []
    progress = Progress(report_every_s=999.0, on_update=lambda e, c, t, m: events.append((e, c, t, m)))
    progress.start(1)
    progress.file_done()

    assert not any(e[0] == "progress" for e in events)
