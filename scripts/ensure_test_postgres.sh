#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/ensure_test_postgres.sh [--dsn <postgres_dsn>] [--pg-home <path>] [--pg-version <major>]

Defaults:
  --dsn         from LSM_TEST_POSTGRES_CONNECTION_STRING
  --pg-home     ~/.local/lsm-pg16
  --pg-version  16

This script is idempotent. It ensures a local PostgreSQL instance is running
for the DSN, creates/updates the DSN user credentials, ensures the database
exists, and enables pgvector.
EOF
}

dsn="${LSM_TEST_POSTGRES_CONNECTION_STRING:-}"
pg_home="${LSM_TEST_PG_HOME:-$HOME/.local/lsm-pg16}"
pg_version="${LSM_TEST_PG_VERSION:-16}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dsn)
      dsn="${2:-}"
      shift 2
      ;;
    --pg-home)
      pg_home="${2:-}"
      shift 2
      ;;
    --pg-version)
      pg_version="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$dsn" ]]; then
  echo "No DSN provided. Set LSM_TEST_POSTGRES_CONNECTION_STRING or pass --dsn." >&2
  exit 1
fi

readarray -t dsn_parts < <(python3 - "$dsn" <<'PY'
import sys
from urllib.parse import urlparse, unquote

dsn = sys.argv[1]
parsed = urlparse(dsn)
if parsed.scheme not in {"postgresql", "postgres"}:
    raise SystemExit(f"Unsupported DSN scheme: {parsed.scheme!r}")

host = (parsed.hostname or "").strip()
port = parsed.port or 5432
database = parsed.path.lstrip("/")
username = parsed.username or ""
password = parsed.password or ""

print(host)
print(port)
print(unquote(database))
print(unquote(username))
print(unquote(password))
PY
)

db_host="${dsn_parts[0]:-}"
db_port="${dsn_parts[1]:-}"
db_name="${dsn_parts[2]:-}"
db_user="${dsn_parts[3]:-}"
db_pass="${dsn_parts[4]:-}"

if [[ -z "$db_host" || -z "$db_port" || -z "$db_name" || -z "$db_user" || -z "$db_pass" ]]; then
  echo "DSN is missing required host/port/database/user/password values." >&2
  exit 1
fi

if [[ "$db_host" != "localhost" && "$db_host" != "127.0.0.1" && "$db_host" != "::1" ]]; then
  echo "Refusing to bootstrap non-local DSN host: $db_host" >&2
  exit 1
fi

bin_dir="/usr/lib/postgresql/${pg_version}/bin"
for exe in initdb pg_ctl pg_isready psql; do
  if [[ ! -x "${bin_dir}/${exe}" ]]; then
    echo "Missing PostgreSQL binary: ${bin_dir}/${exe}" >&2
    exit 1
  fi
done

data_dir="${pg_home}/data"
log_file="${pg_home}/postgres.log"
socket_dir="${pg_home}"

mkdir -p "$pg_home"

if [[ ! -f "${data_dir}/PG_VERSION" ]]; then
  "${bin_dir}/initdb" -D "$data_dir" --auth-local=trust --auth-host=scram-sha-256 -U postgres >/tmp/lsm_pg_initdb.log
fi

if ! "${bin_dir}/pg_ctl" -D "$data_dir" status >/dev/null 2>&1; then
  "${bin_dir}/pg_ctl" -D "$data_dir" -l "$log_file" -o "-p ${db_port} -h localhost -k ${socket_dir}" start
fi

ready=false
for _ in $(seq 1 40); do
  if "${bin_dir}/pg_isready" -h localhost -p "$db_port" -U postgres >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 0.25
done

if [[ "$ready" != "true" ]]; then
  echo "PostgreSQL did not become ready on localhost:${db_port}" >&2
  tail -n 60 "$log_file" >&2 || true
  exit 1
fi

admin_psql=("${bin_dir}/psql" -h "$socket_dir" -p "$db_port" -U postgres -d postgres -v ON_ERROR_STOP=1)

db_user_lit="${db_user//\'/\'\'}"
db_pass_lit="${db_pass//\'/\'\'}"
db_name_lit="${db_name//\'/\'\'}"
db_user_ident="${db_user//\"/\"\"}"
db_name_ident="${db_name//\"/\"\"}"

if ! "${admin_psql[@]}" -tAc "SELECT 1 FROM pg_roles WHERE rolname='${db_user_lit}'" | rg -q "^1$"; then
  "${admin_psql[@]}" -c "CREATE ROLE \"${db_user_ident}\" LOGIN PASSWORD '${db_pass_lit}'"
else
  "${admin_psql[@]}" -c "ALTER ROLE \"${db_user_ident}\" LOGIN PASSWORD '${db_pass_lit}'"
fi

if ! "${admin_psql[@]}" -tAc "SELECT 1 FROM pg_database WHERE datname='${db_name_lit}'" | rg -q "^1$"; then
  "${admin_psql[@]}" -c "CREATE DATABASE \"${db_name_ident}\" OWNER \"${db_user_ident}\""
fi

"${admin_psql[@]}" -c "ALTER DATABASE \"${db_name_ident}\" OWNER TO \"${db_user_ident}\""

"${bin_dir}/psql" -h "$socket_dir" -p "$db_port" -U postgres -d "$db_name" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector"
"${bin_dir}/psql" -h "$socket_dir" -p "$db_port" -U postgres -d "$db_name" -v ON_ERROR_STOP=1 -c "GRANT ALL ON SCHEMA public TO \"${db_user_ident}\""

PGPASSWORD="$db_pass" "${bin_dir}/psql" -h localhost -p "$db_port" -U "$db_user" -d "$db_name" -w -tAc "SELECT current_user, current_database()"

echo "READY user=${db_user} db=${db_name} host=localhost port=${db_port} data=${data_dir}"
