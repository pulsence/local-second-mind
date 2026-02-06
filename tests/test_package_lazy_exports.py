from __future__ import annotations

import importlib

import pytest


def test_lsm_lazy_exports_and_errors() -> None:
    pkg = importlib.import_module("lsm")
    assert getattr(pkg, "ingest", None) is not None
    assert getattr(pkg, "query", None) is not None
    with pytest.raises(AttributeError):
        _ = pkg.not_a_real_export


def test_query_lazy_exports_and_errors() -> None:
    qpkg = importlib.import_module("lsm.query")
    assert callable(qpkg.query)
    assert callable(qpkg.query_sync)
    assert qpkg.QueryResult is not None
    assert qpkg.SessionState is not None
    assert qpkg.Candidate is not None
    assert callable(qpkg.prepare_local_candidates)
    assert qpkg.LocalQueryPlan is not None
    assert qpkg.remote is not None
    with pytest.raises(AttributeError):
        _ = qpkg.not_a_real_export


def test_ingest_lazy_function() -> None:
    ipkg = importlib.import_module("lsm.ingest")
    assert callable(ipkg.ingest)
