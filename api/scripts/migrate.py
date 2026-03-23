"""
api/scripts/migrate.py

Minimal SQL migration runner for Supabase Postgres.

Usage (from project root):
    python -m api.scripts.migrate

How it works:
  1. Connects to Postgres via DATABASE_URL from .env
  2. Creates a `schema_migrations` tracking table if it doesn't exist
  3. Reads every *.sql file in api/schemas/migrations/ in filename order
  4. Skips files already recorded in schema_migrations
  5. Applies the remaining files in a single transaction each
  6. Records each success so re-runs are safe (idempotent)

Future migrations: add api/schemas/migrations/002_whatever.sql and re-run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Bootstrap: load .env before importing anything else ──────────────────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL", "")

if not DATABASE_URL:
    print(
        "\n[migrate] ERROR: DATABASE_URL is not set.\n"
        "  Add it to your .env:\n"
        "    DATABASE_URL=postgresql://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres\n"
        "  Get it from: Supabase dashboard → Settings → Database → Connection string → URI\n"
    )
    sys.exit(1)


try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print(
        "\n[migrate] ERROR: psycopg2-binary is not installed.\n"
        "  Run: pip install psycopg2-binary\n"
        "  Or:  pip install -r api/requirements.txt\n"
    )
    sys.exit(1)


# ── Paths ─────────────────────────────────────────────────────────────────────

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "schemas" / "migrations"


# ── Helpers ───────────────────────────────────────────────────────────────────

_CREATE_TRACKING_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
  filename   text        PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
);
"""


def _already_applied(cur, filename: str) -> bool:
    cur.execute("SELECT 1 FROM schema_migrations WHERE filename = %s", (filename,))
    return cur.fetchone() is not None


def _record_migration(cur, filename: str) -> None:
    cur.execute(
        "INSERT INTO schema_migrations (filename) VALUES (%s) ON CONFLICT DO NOTHING",
        (filename,),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not sql_files:
        print("[migrate] No migration files found in", MIGRATIONS_DIR)
        return

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            # Ensure tracking table exists
            cur.execute(_CREATE_TRACKING_TABLE)
            conn.commit()

        applied = 0
        skipped = 0

        for path in sql_files:
            fname = path.name
            with conn.cursor() as cur:
                if _already_applied(cur, fname):
                    print(f"[migrate] skip   {fname}  (already applied)")
                    skipped += 1
                    continue

                print(f"[migrate] apply  {fname} ...", end=" ", flush=True)
                sql = path.read_text(encoding="utf-8")
                cur.execute(sql)
                _record_migration(cur, fname)
                conn.commit()
                print("ok")
                applied += 1

        print(f"\n[migrate] Done — {applied} applied, {skipped} skipped.")

    except Exception as exc:
        conn.rollback()
        print(f"\n[migrate] FAILED: {exc}")
        sys.exit(1)

    finally:
        conn.close()


if __name__ == "__main__":
    run()
