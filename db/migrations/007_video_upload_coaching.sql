-- ──────────────────────────────────────────────────────────────────────────────
-- Migration 007: Climber Video Upload + Coaching Sessions
--
-- Purpose:
--   Allow climbers to upload their own videos, run them through the pose
--   extractor pipeline, and receive automated coaching insights. Videos are
--   processed server-side; only the analysis results (not the video file) are
--   stored permanently.
--
-- Tables:
--   video_upload_sessions    Tracks each upload: file metadata, status, route link
--   coaching_insights        Structured per-insight rows from video_coach.py
-- ──────────────────────────────────────────────────────────────────────────────

-- ── Upload sessions ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS video_upload_sessions (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Who uploaded (optional — null for anonymous)
    climber_id      UUID        REFERENCES climber_profiles(id) ON DELETE SET NULL,

    -- File info (video is deleted after processing; path is tmp only)
    original_filename TEXT,
    file_size_bytes   BIGINT,
    duration_sec      FLOAT,
    fps               FLOAT,
    resolution        TEXT,                 -- e.g. '1080x1920'

    -- Route context (optional — helps coaching benchmarks choose correct grade)
    route_id        UUID        REFERENCES board_routes(id) ON DELETE SET NULL,
    board_angle_deg INT,
    self_reported_grade TEXT,              -- climber's guess e.g. 'V6'

    -- Processing state
    -- pending → processing → done | failed
    status          TEXT        NOT NULL DEFAULT 'pending',
    error_message   TEXT,
    frames_extracted INT,
    processed_at    TIMESTAMP,

    -- Coaching summary (top-level result from video_coach.analyse_attempt)
    overall_verdict TEXT,                  -- 'good' | 'needs_work' | 'critical'
    summary_text    TEXT,
    attempt_id      TEXT,                  -- UUID used in pose_frames for this upload

    created_at      TIMESTAMP   DEFAULT NOW()
);

COMMENT ON TABLE video_upload_sessions IS
    'One row per climber video upload. Video file is deleted after pose extraction.';

CREATE INDEX IF NOT EXISTS idx_uploads_climber ON video_upload_sessions (climber_id);
CREATE INDEX IF NOT EXISTS idx_uploads_status  ON video_upload_sessions (status);


-- ── Per-insight rows ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS coaching_insights (
    id              SERIAL      PRIMARY KEY,
    session_id      UUID        NOT NULL REFERENCES video_upload_sessions(id) ON DELETE CASCADE,

    category        TEXT        NOT NULL,  -- tension | arms | hips | sequencing | footwork
    severity        TEXT        NOT NULL,  -- high | medium | low
    message         TEXT        NOT NULL,  -- human-readable description
    score           FLOAT,                 -- measured metric value
    benchmark       TEXT,                  -- expected range string e.g. '0.50–0.68'
    drills          TEXT[],                -- array of drill descriptions

    created_at      TIMESTAMP   DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_insights_session  ON coaching_insights (session_id);
CREATE INDEX IF NOT EXISTS idx_insights_category ON coaching_insights (category, severity);
