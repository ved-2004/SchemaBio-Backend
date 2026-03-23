-- Migration 001: Full initial schema — all 6 tables
-- This is the source of truth for a fresh Supabase project.
-- Tables are created in dependency order (referenced tables first).
-- All user-owned tables cascade-delete when the parent user row is deleted.


-- ── 1. users ─────────────────────────────────────────────────────────────────
-- Populated on every Google OAuth login. id is the platform UUID for all FKs.

CREATE TABLE IF NOT EXISTS users (
  id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  google_id     text        NOT NULL UNIQUE,
  email         text        NOT NULL,
  name          text        NOT NULL,
  avatar_url    text,
  phone_number  text,
  created_at    timestamptz NOT NULL DEFAULT now(),
  last_login_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS users_google_id_idx ON users(google_id);
CREATE UNIQUE INDEX IF NOT EXISTS users_email_idx     ON users(email);


-- ── 2. user_uploads ──────────────────────────────────────────────────────────
-- File metadata for Supabase Storage uploads. user_id is nullable so the row
-- can exist even if an anonymous upload is recorded (no FK target required).
-- Authenticated uploads carry a uuid FK; anonymous uploads leave it NULL.

CREATE TABLE IF NOT EXISTS user_uploads (
  id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  upload_id       text        NOT NULL UNIQUE,
  user_id         uuid        REFERENCES users(id) ON DELETE CASCADE,
  filename        text        NOT NULL,
  file_size_bytes bigint      NOT NULL DEFAULT 0,
  bucket_path     text        NOT NULL,
  program_id      text,
  uploaded_at     timestamptz NOT NULL DEFAULT now(),
  expires_at      timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS user_uploads_user_id_idx    ON user_uploads(user_id);
CREATE INDEX IF NOT EXISTS user_uploads_expires_at_idx ON user_uploads(expires_at);


-- ── 3. programs ──────────────────────────────────────────────────────────────
-- Layer 1 DrugProgram output stored as JSONB. user_id is text (legacy; not FK)
-- so anonymous /api/analyze calls still persist their results.

CREATE TABLE IF NOT EXISTS programs (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  program_id  text        NOT NULL UNIQUE,
  user_id     text,
  data        jsonb       NOT NULL DEFAULT '{}'::jsonb,
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS programs_program_id_idx ON programs(program_id);
CREATE INDEX IF NOT EXISTS programs_user_id_idx    ON programs(user_id);


-- ── 4. experiment_runs ───────────────────────────────────────────────────────
-- One row per pipeline session (created by /api/upload-and-parse for auth users).
-- program_id is set immediately after synchronous ingestion.
-- upload_ids is the array of user_uploads.upload_id values used in the run.

CREATE TABLE IF NOT EXISTS experiment_runs (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     uuid        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  program_id  text,
  upload_ids  text[]      NOT NULL DEFAULT '{}',
  status      text        NOT NULL DEFAULT 'complete',  -- complete | failed
  error       text,
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS experiment_runs_user_id_idx    ON experiment_runs(user_id);
CREATE INDEX IF NOT EXISTS experiment_runs_program_id_idx ON experiment_runs(program_id);
CREATE INDEX IF NOT EXISTS experiment_runs_created_at_idx ON experiment_runs(created_at DESC);


-- ── 5. experiment_results ────────────────────────────────────────────────────
-- Layer 2 output (ranked experiments, hypotheses, bioinf tasks, etc.)
-- Unique per run; re-running Layer 2 on the same run upserts on run_id.

CREATE TABLE IF NOT EXISTS experiment_results (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id      uuid        NOT NULL REFERENCES experiment_runs(id) ON DELETE CASCADE,
  user_id     uuid        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  program_id  text        NOT NULL,
  data        jsonb       NOT NULL DEFAULT '{}'::jsonb,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS experiment_results_run_id_idx  ON experiment_results(run_id);
CREATE        INDEX IF NOT EXISTS experiment_results_user_id_idx ON experiment_results(user_id);


-- ── 6. execution_plans ───────────────────────────────────────────────────────
-- Layer 3 output (FDA pathway, CRO routing, grants, readiness scores, etc.)
-- Unique per run; re-running Layer 3 on the same run upserts on run_id.

CREATE TABLE IF NOT EXISTS execution_plans (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id      uuid        NOT NULL REFERENCES experiment_runs(id) ON DELETE CASCADE,
  user_id     uuid        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  program_id  text        NOT NULL,
  data        jsonb       NOT NULL DEFAULT '{}'::jsonb,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS execution_plans_run_id_idx  ON execution_plans(run_id);
CREATE        INDEX IF NOT EXISTS execution_plans_user_id_idx ON execution_plans(user_id);
