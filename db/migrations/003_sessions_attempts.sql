-- Migration 003: sessions and attempts
-- Sessions track a climber's visit to the board.
-- Attempts track each individual route try within a session.

-- ─────────────────────────────────────────
-- SESSIONS
-- One row per board visit.
-- A session ties a climber to a gym on a
-- specific day, at a specific board angle.
-- Video is stored as a path to the raw file;
-- processing happens asynchronously.
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    climber_id      UUID NOT NULL REFERENCES climbers (id) ON DELETE CASCADE,
    gym_id          UUID NOT NULL REFERENCES gyms (id) ON DELETE CASCADE,

    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,

    -- The board angle used during this session
    board_angle_deg INT NOT NULL CHECK (board_angle_deg BETWEEN 0 AND 70),
    board_type      TEXT NOT NULL,  -- 'kilter' | 'moonboard' | 'custom'

    -- Path to raw session video (if captured).
    -- Relative to the media storage root.
    -- NULL if no video was recorded.
    video_path      TEXT,

    -- Processing status for the video pipeline
    -- 'pending' → 'processing' → 'done' | 'failed'
    video_status    TEXT NOT NULL DEFAULT 'none'
        CHECK (video_status IN ('none', 'pending', 'processing', 'done', 'failed')),

    -- Free-form notes (coach observations, climber journal, etc.)
    notes           TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_sessions_climber_id ON sessions (climber_id);
CREATE INDEX idx_sessions_gym_id     ON sessions (gym_id);
CREATE INDEX idx_sessions_started_at ON sessions (started_at DESC);
CREATE INDEX idx_sessions_video_status ON sessions (video_status)
    WHERE video_status IN ('pending', 'processing');


-- ─────────────────────────────────────────
-- ATTEMPTS
-- One row per route try within a session.
-- An attempt_number > 1 means the climber
-- tried the same route multiple times.
-- completed = TRUE means they sent it.
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS attempts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    route_id        UUID NOT NULL REFERENCES board_routes (id),
    climber_id      UUID NOT NULL REFERENCES climbers (id),

    -- Which try this is on this route within this session
    attempt_number  INT NOT NULL DEFAULT 1 CHECK (attempt_number >= 1),

    completed       BOOLEAN NOT NULL DEFAULT FALSE,
    -- How long the climber was on the wall for this attempt (seconds)
    duration_sec    FLOAT CHECK (duration_sec >= 0),

    -- Timestamp offset within the session video (seconds from video start).
    -- Used to slice the relevant video segment for pose processing.
    video_offset_sec    FLOAT,
    video_end_sec       FLOAT,

    attempted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_attempts_session_id  ON attempts (session_id);
CREATE INDEX idx_attempts_route_id    ON attempts (route_id);
CREATE INDEX idx_attempts_climber_id  ON attempts (climber_id);
CREATE INDEX idx_attempts_completed   ON attempts (climber_id, completed);

-- Convenience: prevent logging the same attempt_number twice per route/session
CREATE UNIQUE INDEX idx_attempts_unique_try
    ON attempts (session_id, route_id, attempt_number);
