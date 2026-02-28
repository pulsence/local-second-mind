"""
SQLite + sqlite-vec provider implementation.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from struct import unpack
from typing import Any, Dict, List, Optional, Tuple

from lsm.config.models import VectorDBConfig
from lsm.db.connection import create_sqlite_connection, resolve_db_path
from lsm.db.schema import APPLICATION_TABLES, ensure_application_schema
from lsm.db.transaction import transaction
from lsm.logging import get_logger

from .base import BaseVectorDBProvider, PruneCriteria, VectorDBGetResult, VectorDBQueryResult

logger = get_logger(__name__)


def _to_bool_int(value: Any, *, default: int = 1) -> int:
    if value is None:
        return int(default)
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) != 0 else 0
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return 1
    if normalized in {"0", "false", "no", "off"}:
        return 0
    return int(default)


def _normalize_ext(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text.startswith("."):
        return text
    return f".{text}"


class SQLiteVecProvider(BaseVectorDBProvider):
    """Vector DB provider backed by SQLite + sqlite-vec."""

    def __init__(self, config: VectorDBConfig) -> None:
        super().__init__(config)
        self._db_path = resolve_db_path(config.path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = create_sqlite_connection(self._db_path)

        self._extension_loaded = False
        self._load_extension()
        self._ensure_schema()

    @property
    def name(self) -> str:
        return "sqlite"

    @property
    def connection(self) -> sqlite3.Connection:
        """Expose the underlying SQLite connection for subsystem reuse."""
        return self._conn

    def _load_extension(self) -> None:
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._extension_loaded = True
        except Exception as exc:
            self._extension_loaded = False
            logger.warning("Failed to load sqlite-vec extension: %s", exc)

    def _ensure_schema(self) -> None:
        if not self._extension_loaded:
            return

        # Application tables (lsm_chunks, lsm_manifest, etc.) — owned by lsm.db.
        ensure_application_schema(self._conn)

        # Vector-specific virtual tables and FTS content-sync triggers.
        self._conn.executescript(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[384] distance_metric=cosine,
                is_current INTEGER,
                node_type TEXT,
                source_path TEXT,
                cluster_id INTEGER
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                chunk_text,
                heading,
                source_name,
                content='lsm_chunks',
                content_rowid='rowid'
            );
            """
        )

        self._conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS lsm_chunks_ai
            AFTER INSERT ON lsm_chunks
            BEGIN
                INSERT INTO chunks_fts(rowid, chunk_id, chunk_text, heading, source_name)
                VALUES (new.rowid, new.chunk_id, new.chunk_text, new.heading, new.source_name);
            END;

            CREATE TRIGGER IF NOT EXISTS lsm_chunks_ad
            AFTER DELETE ON lsm_chunks
            BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, chunk_text, heading, source_name)
                VALUES('delete', old.rowid, old.chunk_id, old.chunk_text, old.heading, old.source_name);
            END;

            CREATE TRIGGER IF NOT EXISTS lsm_chunks_au
            AFTER UPDATE ON lsm_chunks
            BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, chunk_id, chunk_text, heading, source_name)
                VALUES('delete', old.rowid, old.chunk_id, old.chunk_text, old.heading, old.source_name);
                INSERT INTO chunks_fts(rowid, chunk_id, chunk_text, heading, source_name)
                VALUES (new.rowid, new.chunk_id, new.chunk_text, new.heading, new.source_name);
            END;
            """
        )
        self._conn.commit()

    def is_available(self) -> bool:
        return bool(self._extension_loaded)

    def add_chunks(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        if not ids:
            return
        if not self._extension_loaded:
            raise RuntimeError("sqlite-vec extension is not loaded")
        if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
            raise ValueError("ids, documents, metadatas, and embeddings must have the same length")

        import sqlite_vec

        with transaction(self._conn):
            for chunk_id, text, metadata, embedding in zip(ids, documents, metadatas, embeddings):
                normalized = self._normalize_metadata(metadata or {})
                self._conn.execute(
                    """
                    INSERT INTO lsm_chunks (
                        chunk_id, source_path, source_name, chunk_text, heading, heading_path,
                        page_number, paragraph_index, mtime_ns, file_hash, version, is_current,
                        node_type, root_tags, folder_tags, content_type, cluster_id, cluster_size,
                        simhash, ext, chunk_index, ingested_at, start_char, end_char,
                        chunk_length, metadata_json
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        source_path=excluded.source_path,
                        source_name=excluded.source_name,
                        chunk_text=excluded.chunk_text,
                        heading=excluded.heading,
                        heading_path=excluded.heading_path,
                        page_number=excluded.page_number,
                        paragraph_index=excluded.paragraph_index,
                        mtime_ns=excluded.mtime_ns,
                        file_hash=excluded.file_hash,
                        version=excluded.version,
                        is_current=excluded.is_current,
                        node_type=excluded.node_type,
                        root_tags=excluded.root_tags,
                        folder_tags=excluded.folder_tags,
                        content_type=excluded.content_type,
                        cluster_id=excluded.cluster_id,
                        cluster_size=excluded.cluster_size,
                        simhash=excluded.simhash,
                        ext=excluded.ext,
                        chunk_index=excluded.chunk_index,
                        ingested_at=excluded.ingested_at,
                        start_char=excluded.start_char,
                        end_char=excluded.end_char,
                        chunk_length=excluded.chunk_length,
                        metadata_json=excluded.metadata_json
                    """,
                    (
                        str(chunk_id),
                        normalized["source_path"],
                        normalized["source_name"],
                        text or "",
                        normalized["heading"],
                        normalized["heading_path"],
                        normalized["page_number"],
                        normalized["paragraph_index"],
                        normalized["mtime_ns"],
                        normalized["file_hash"],
                        normalized["version"],
                        normalized["is_current"],
                        normalized["node_type"],
                        normalized["root_tags"],
                        normalized["folder_tags"],
                        normalized["content_type"],
                        normalized["cluster_id"],
                        normalized["cluster_size"],
                        normalized["simhash"],
                        normalized["ext"],
                        normalized["chunk_index"],
                        normalized["ingested_at"],
                        normalized["start_char"],
                        normalized["end_char"],
                        normalized["chunk_length"],
                        json.dumps(metadata or {}, ensure_ascii=True),
                    ),
                )
                self._conn.execute(
                    "DELETE FROM vec_chunks WHERE chunk_id = ?",
                    (str(chunk_id),),
                )
                self._conn.execute(
                    """
                    INSERT INTO vec_chunks (
                        chunk_id, embedding, is_current, node_type, source_path, cluster_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(chunk_id),
                        sqlite_vec.serialize_float32(list(embedding)),
                        normalized["is_current"],
                        normalized["node_type"],
                        normalized["source_path"],
                        normalized["cluster_id"],
                    ),
                )

    def get(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0,
        include: Optional[List[str]] = None,
    ) -> VectorDBGetResult:
        if not self._extension_loaded:
            return VectorDBGetResult()

        inc = include or ["metadatas"]
        columns = ["chunk_id", "metadata_json"]
        if "documents" in inc:
            columns.append("chunk_text")
        if "embeddings" in inc:
            columns.append("embedding")

        where: List[str] = []
        params: List[Any] = []

        if ids is not None:
            placeholders = ", ".join(["?"] * len(ids))
            where.append(f"c.chunk_id IN ({placeholders})")
            params.extend([str(item) for item in ids])

        if filters:
            filter_sql, filter_params = self._sql_filters(filters, alias="c")
            if filter_sql:
                where.append(filter_sql)
                params.extend(filter_params)

        query = f"SELECT {', '.join(columns)} FROM lsm_chunks c"
        if "embeddings" in inc:
            query += " LEFT JOIN vec_chunks v ON v.chunk_id = c.chunk_id"
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY c.rowid"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, int(limit)))
        if offset:
            query += " OFFSET ?"
            params.append(max(0, int(offset)))

        rows = self._conn.execute(query, params).fetchall()
        results = VectorDBGetResult(ids=[str(row["chunk_id"]) for row in rows])
        if "documents" in inc:
            results.documents = [str(row["chunk_text"] or "") for row in rows]
        if "metadatas" in inc:
            results.metadatas = [self._metadata_from_row(row) for row in rows]
        if "embeddings" in inc:
            results.embeddings = [self._deserialize_embedding(row["embedding"]) for row in rows]
        return results

    def query(
        self,
        embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> VectorDBQueryResult:
        if not embedding or top_k <= 0:
            return VectorDBQueryResult(ids=[], documents=[], metadatas=[], distances=[])
        if not self._extension_loaded:
            return VectorDBQueryResult(ids=[], documents=[], metadatas=[], distances=[])

        import sqlite_vec

        knn_limit = max(int(top_k), int(top_k) * 10)
        rows = self._conn.execute(
            """
            SELECT chunk_id, distance
            FROM vec_chunks
            WHERE embedding MATCH ? AND k = ?
            """,
            (sqlite_vec.serialize_float32(list(embedding)), knn_limit),
        ).fetchall()
        if not rows:
            return VectorDBQueryResult(ids=[], documents=[], metadatas=[], distances=[])

        candidates: List[Tuple[str, float]] = [
            (str(row["chunk_id"]), float(row["distance"])) for row in rows
        ]
        ids = [item[0] for item in candidates]
        placeholders = ", ".join(["?"] * len(ids))
        chunk_rows = self._conn.execute(
            f"""
            SELECT chunk_id, chunk_text, metadata_json
            FROM lsm_chunks
            WHERE chunk_id IN ({placeholders})
            """,
            ids,
        ).fetchall()
        metadata_by_id = {
            str(row["chunk_id"]): (
                str(row["chunk_text"] or ""),
                self._metadata_from_row(row),
            )
            for row in chunk_rows
        }

        result_ids: List[str] = []
        result_docs: List[str] = []
        result_metas: List[Dict[str, Any]] = []
        result_dists: List[Optional[float]] = []

        for chunk_id, distance in candidates:
            if chunk_id not in metadata_by_id:
                continue
            text, metadata = metadata_by_id[chunk_id]
            if filters and not self._match_metadata_filter(metadata, filters):
                continue
            result_ids.append(chunk_id)
            result_docs.append(text)
            result_metas.append(metadata)
            result_dists.append(distance)
            if len(result_ids) >= int(top_k):
                break

        return VectorDBQueryResult(
            ids=result_ids,
            documents=result_docs,
            metadatas=result_metas,
            distances=result_dists,
        )

    def delete_by_id(self, ids: List[str]) -> None:
        if not ids:
            return
        placeholders = ", ".join(["?"] * len(ids))
        normalized = [str(item) for item in ids]
        with transaction(self._conn):
            self._conn.execute(
                f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})",
                normalized,
            )
            self._conn.execute(
                f"DELETE FROM lsm_chunks WHERE chunk_id IN ({placeholders})",
                normalized,
            )

    def delete_by_filter(self, filters: Dict[str, Any]) -> None:
        if not filters:
            raise ValueError("filters must be a non-empty dict")
        where_sql, params = self._sql_filters(filters, alias="c")
        query = "SELECT c.chunk_id, c.metadata_json FROM lsm_chunks c"
        if where_sql:
            query += f" WHERE {where_sql}"
        rows = self._conn.execute(query, params).fetchall()
        selected = [
            str(row["chunk_id"])
            for row in rows
            if self._match_metadata_filter(self._metadata_from_row(row), filters)
        ]
        self.delete_by_id(selected)

    def delete_all(self) -> int:
        current = self.count()
        with transaction(self._conn):
            self._conn.execute("DELETE FROM vec_chunks")
            self._conn.execute("DELETE FROM lsm_chunks")
        return current

    def count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS total FROM lsm_chunks WHERE is_current = 1"
        ).fetchone()
        return int((row["total"] if row else 0) or 0)

    def get_stats(self) -> Dict[str, Any]:
        row = self._conn.execute(
            """
            SELECT
                COUNT(*) AS total_rows,
                SUM(CASE WHEN is_current = 1 THEN 1 ELSE 0 END) AS current_rows,
                COUNT(DISTINCT source_path) AS unique_sources
            FROM lsm_chunks
            """
        ).fetchone()
        return {
            "provider": self.name,
            "database": str(self._db_path),
            "count": int((row["current_rows"] if row else 0) or 0),
            "total_rows": int((row["total_rows"] if row else 0) or 0),
            "unique_sources": int((row["unique_sources"] if row else 0) or 0),
        }

    def optimize(self) -> Dict[str, Any]:
        self._conn.execute("VACUUM")
        self._conn.execute("ANALYZE")
        return {"provider": self.name, "status": "ok"}

    def health_check(self) -> Dict[str, Any]:
        if not self._extension_loaded:
            return {
                "provider": self.name,
                "status": "error",
                "error": "sqlite-vec extension is not loaded",
            }

        required = list(APPLICATION_TABLES) + ["vec_chunks", "chunks_fts"]
        missing = []
        for name in required:
            row = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE name = ?",
                (name,),
            ).fetchone()
            if row is None:
                missing.append(name)

        if missing:
            return {
                "provider": self.name,
                "status": "error",
                "error": f"missing tables: {', '.join(sorted(missing))}",
            }
        return {"provider": self.name, "status": "ok", "count": self.count()}

    def update_metadatas(self, ids: List[str], metadatas: List[Dict[str, Any]]) -> None:
        if not ids:
            return
        if len(ids) != len(metadatas):
            raise ValueError("ids and metadatas must have the same length")

        with transaction(self._conn):
            for chunk_id, metadata in zip(ids, metadatas):
                normalized = self._normalize_metadata(metadata or {})
                self._conn.execute(
                    """
                    UPDATE lsm_chunks
                    SET
                        source_path = COALESCE(?, source_path),
                        source_name = COALESCE(?, source_name),
                        heading = COALESCE(?, heading),
                        heading_path = COALESCE(?, heading_path),
                        page_number = COALESCE(?, page_number),
                        paragraph_index = COALESCE(?, paragraph_index),
                        mtime_ns = COALESCE(?, mtime_ns),
                        file_hash = COALESCE(?, file_hash),
                        version = COALESCE(?, version),
                        is_current = ?,
                        node_type = COALESCE(?, node_type),
                        root_tags = COALESCE(?, root_tags),
                        folder_tags = COALESCE(?, folder_tags),
                        content_type = COALESCE(?, content_type),
                        cluster_id = COALESCE(?, cluster_id),
                        cluster_size = COALESCE(?, cluster_size),
                        simhash = COALESCE(?, simhash),
                        ext = COALESCE(?, ext),
                        chunk_index = COALESCE(?, chunk_index),
                        ingested_at = COALESCE(?, ingested_at),
                        start_char = COALESCE(?, start_char),
                        end_char = COALESCE(?, end_char),
                        chunk_length = COALESCE(?, chunk_length),
                        metadata_json = ?
                    WHERE chunk_id = ?
                    """,
                    (
                        normalized["source_path"],
                        normalized["source_name"],
                        normalized["heading"],
                        normalized["heading_path"],
                        normalized["page_number"],
                        normalized["paragraph_index"],
                        normalized["mtime_ns"],
                        normalized["file_hash"],
                        normalized["version"],
                        normalized["is_current"],
                        normalized["node_type"],
                        normalized["root_tags"],
                        normalized["folder_tags"],
                        normalized["content_type"],
                        normalized["cluster_id"],
                        normalized["cluster_size"],
                        normalized["simhash"],
                        normalized["ext"],
                        normalized["chunk_index"],
                        normalized["ingested_at"],
                        normalized["start_char"],
                        normalized["end_char"],
                        normalized["chunk_length"],
                        json.dumps(metadata or {}, ensure_ascii=True),
                        str(chunk_id),
                    ),
                )
                self._conn.execute(
                    """
                    UPDATE vec_chunks
                    SET
                        is_current = ?,
                        node_type = ?,
                        source_path = ?,
                        cluster_id = ?
                    WHERE chunk_id = ?
                    """,
                    (
                        normalized["is_current"],
                        normalized["node_type"],
                        normalized["source_path"],
                        normalized["cluster_id"],
                        str(chunk_id),
                    ),
                )

    def prune_old_versions(self, criteria: PruneCriteria) -> int:
        max_versions = (
            int(criteria.max_versions)
            if criteria.max_versions is not None
            else None
        )
        older_than_days = (
            int(criteria.older_than_days)
            if criteria.older_than_days is not None
            else None
        )

        if max_versions is not None and max_versions < 1:
            raise ValueError("max_versions must be >= 1 when provided")
        if older_than_days is not None and older_than_days < 0:
            raise ValueError("older_than_days must be >= 0 when provided")

        rows = self._conn.execute(
            """
            SELECT chunk_id, source_path, version, ingested_at
            FROM lsm_chunks
            WHERE is_current = 0
            """
        ).fetchall()
        if not rows:
            return 0

        keep_versions_by_source: dict[str, set[int]] = {}
        if max_versions is not None:
            versions_by_source: dict[str, set[int]] = {}
            for row in rows:
                source_path = str(row["source_path"] or "")
                version = int(row["version"] or 0)
                versions_by_source.setdefault(source_path, set()).add(version)
            for source_path, versions in versions_by_source.items():
                keep_versions_by_source[source_path] = set(
                    sorted(versions, reverse=True)[:max_versions]
                )

        cutoff: Optional[datetime] = None
        if older_than_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)

        to_delete: List[str] = []
        for row in rows:
            source_path = str(row["source_path"] or "")
            version = int(row["version"] or 0)
            ingested_at = self._parse_iso_datetime(row["ingested_at"])

            if max_versions is not None:
                kept = keep_versions_by_source.get(source_path, set())
                if version in kept:
                    continue

            if cutoff is not None:
                # If timestamp is unavailable, keep the row for safety.
                if ingested_at is None or ingested_at > cutoff:
                    continue

            to_delete.append(str(row["chunk_id"]))

        if not to_delete:
            return 0

        with transaction(self._conn):
            placeholders = ", ".join(["?"] * len(to_delete))
            self._conn.execute(
                f"DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})",
                to_delete,
            )
            self._conn.execute(
                f"DELETE FROM lsm_chunks WHERE chunk_id IN ({placeholders})",
                to_delete,
            )
        return len(to_delete)

    @staticmethod
    def _normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        source_path = str(metadata.get("source_path", "") or "")
        if not source_path:
            raise ValueError("metadata.source_path is required")
        cluster_id_raw = metadata.get("cluster_id")
        cluster_id = int(cluster_id_raw) if cluster_id_raw is not None else 0
        return {
            "source_path": source_path,
            "source_name": metadata.get("source_name"),
            "heading": metadata.get("heading"),
            "heading_path": metadata.get("heading_path"),
            "page_number": metadata.get("page_number"),
            "paragraph_index": metadata.get("paragraph_index"),
            "mtime_ns": metadata.get("mtime_ns"),
            "file_hash": metadata.get("file_hash"),
            "version": metadata.get("version"),
            "is_current": _to_bool_int(metadata.get("is_current"), default=1),
            "node_type": str(metadata.get("node_type", "chunk") or "chunk"),
            "root_tags": metadata.get("root_tags"),
            "folder_tags": metadata.get("folder_tags"),
            "content_type": metadata.get("content_type"),
            "cluster_id": cluster_id,
            "cluster_size": metadata.get("cluster_size"),
            "simhash": metadata.get("simhash"),
            "ext": _normalize_ext(metadata.get("ext")),
            "chunk_index": metadata.get("chunk_index"),
            "ingested_at": metadata.get("ingested_at"),
            "start_char": metadata.get("start_char"),
            "end_char": metadata.get("end_char"),
            "chunk_length": metadata.get("chunk_length"),
        }

    @staticmethod
    def _parse_iso_datetime(raw: Any) -> Optional[datetime]:
        text = str(raw or "").strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _deserialize_embedding(raw: Any) -> List[float]:
        if raw is None:
            return []
        blob = bytes(raw)
        if not blob:
            return []
        dimension = len(blob) // 4
        return list(unpack(f"{dimension}f", blob))

    @staticmethod
    def _metadata_from_row(row: sqlite3.Row) -> Dict[str, Any]:
        try:
            raw = row["metadata_json"]
            parsed = json.loads(str(raw or "{}"))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {}

    def _sql_filters(self, filters: Dict[str, Any], *, alias: str = "") -> Tuple[str, List[Any]]:
        if not filters:
            return "", []
        prefix = f"{alias}." if alias else ""
        clauses: List[str] = []
        params: List[Any] = []

        for key, value in filters.items():
            normalized_value = self._normalize_filter_value(value)
            if key == "source_path":
                clauses.append(f"{prefix}source_path = ?")
                params.append(str(normalized_value))
            elif key == "source_name":
                clauses.append(f"{prefix}source_name = ?")
                params.append(str(normalized_value))
            elif key == "ext":
                clauses.append(f"{prefix}ext = ?")
                params.append(_normalize_ext(normalized_value))
            elif key == "is_current":
                clauses.append(f"{prefix}is_current = ?")
                params.append(_to_bool_int(normalized_value))
            elif key == "node_type":
                clauses.append(f"{prefix}node_type = ?")
                params.append(str(normalized_value))
            elif key == "file_hash":
                clauses.append(f"{prefix}file_hash = ?")
                params.append(str(normalized_value))
            elif key == "version":
                clauses.append(f"{prefix}version = ?")
                params.append(int(normalized_value))
            elif key == "path_contains":
                path_values = normalized_value if isinstance(normalized_value, list) else [normalized_value]
                path_values = [str(item).strip() for item in path_values if str(item).strip()]
                if path_values:
                    path_clauses = [f"LOWER({prefix}source_path) LIKE ?" for _ in path_values]
                    clauses.append("(" + " OR ".join(path_clauses) + ")")
                    params.extend([f"%{item.lower()}%" for item in path_values])
            elif key == "ext_allow":
                allowed = [_normalize_ext(item) for item in self._ensure_list(normalized_value)]
                allowed = [item for item in allowed if item]
                if allowed:
                    placeholders = ", ".join(["?"] * len(allowed))
                    clauses.append(f"{prefix}ext IN ({placeholders})")
                    params.extend(allowed)
            elif key == "ext_deny":
                denied = [_normalize_ext(item) for item in self._ensure_list(normalized_value)]
                denied = [item for item in denied if item]
                if denied:
                    placeholders = ", ".join(["?"] * len(denied))
                    clauses.append(f"({prefix}ext IS NULL OR {prefix}ext NOT IN ({placeholders}))")
                    params.extend(denied)

        return (" AND ".join(clauses), params)

    @staticmethod
    def _normalize_filter_value(value: Any) -> Any:
        if isinstance(value, dict) and len(value) == 1:
            op, inner = next(iter(value.items()))
            if op == "$eq":
                return inner
        return value

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        return [value]

    def _match_metadata_filter(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        if not filters:
            return True
        for key, raw_value in filters.items():
            value = self._normalize_filter_value(raw_value)
            if key == "path_contains":
                source_path = str(metadata.get("source_path", "") or "").lower()
                values = value if isinstance(value, list) else [value]
                needles = [str(item).strip().lower() for item in values if str(item).strip()]
                if needles and not any(needle in source_path for needle in needles):
                    return False
                continue
            if key == "ext_allow":
                ext = _normalize_ext(metadata.get("ext"))
                allowed = {_normalize_ext(item) for item in self._ensure_list(value)}
                allowed = {item for item in allowed if item}
                if allowed and ext not in allowed:
                    return False
                continue
            if key == "ext_deny":
                ext = _normalize_ext(metadata.get("ext"))
                denied = {_normalize_ext(item) for item in self._ensure_list(value)}
                denied = {item for item in denied if item}
                if denied and ext in denied:
                    return False
                continue
            if key == "is_current":
                if _to_bool_int(metadata.get("is_current"), default=1) != _to_bool_int(value, default=1):
                    return False
                continue

            meta_value = metadata.get(key)
            if isinstance(value, list):
                if meta_value not in value:
                    return False
            else:
                if meta_value != value:
                    return False
        return True
