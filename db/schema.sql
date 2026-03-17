-- ============================================================
-- schema.sql — Full database schema for Climbing Intelligence Platform
-- Generated from migrations 001–005.
-- Run this file to create the entire schema from scratch:
--   psql climbing_platform < db/schema.sql
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ────────────────────────────────────────────────────────────
-- GYMS
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gyms (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name                TEXT NOT NULL,
    location            TEXT,
    kilter_board_angle  INT CHECK (kilter_board_angle BETWEEN 0 AND 70),
    has_moon_board      BOOLEAN NOT NULL DEFAULT FALSE,
    has_kilter_board    BOOLEAN NOT NULL DEFAULT FALSE,
    contact_email       TEXT,
    timezone            TEXT NOT NULL DEFAULT 'America/Chicago',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_gyms_name ON gyms (name);

-- ────────────────────────────────────────────────────────────
-- CLIMBERS
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS climbers (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    gym_id                  UUID NOT NULL REFERENCES gyms (id) ON DELETE CASCADE,
    name                    TEXT NOT NULL,
    email                   TEXT UNIQUE,
    self_reported_grade     TEXT,
    goals                   TEXT[] NOT NULL DEFAULT '{}',
    weak_move_types         TEXT[] NOT NULL DEFAULT '{}',
    years_climbing          INT CHECK (years_climbing >= 0),
    height_cm               INT,
    wingspan_cm             INT,
    video_consent           BOOLEAN NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_climbers_gym_id ON climbers (gym_id);
CREATE INDEX idx_climbers_grade  ON climbers (self_reported_grade);

-- ────────────────────────────────────────────────────────────
-- BOARD_ROUTES
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS board_routes (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source              TEXT NOT NULL CHECK (source IN ('kilter', 'moonboard', 'custom')),
    external_id         TEXT,
    board_type          TEXT NOT NULL,
    board_angle_deg     INT NOT NULL CHECK (board_angle_deg BETWEEN 0 AND 70),
    name                TEXT,
    setter_name         TEXT,
    community_grade     TEXT,
    difficulty_score    FLOAT CHECK (difficulty_score BETWEEN 0.0 AND 1.0),
    send_count          INT NOT NULL DEFAULT 0,
    avg_quality_rating  FLOAT,
    metadata            JSONB NOT NULL DEFAULT '{}',
    synced_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX idx_board_routes_external    ON board_routes (source, external_id) WHERE external_id IS NOT NULL;
CREATE INDEX idx_board_routes_source             ON board_routes (source);
CREATE INDEX idx_board_routes_grade              ON board_routes (community_grade);
CREATE INDEX idx_board_routes_difficulty         ON board_routes (difficulty_score);
CREATE INDEX idx_board_routes_send_count         ON board_routes (send_count DESC);

-- ────────────────────────────────────────────────────────────
-- ROUTE_HOLDS
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS route_holds (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    route_id        UUID NOT NULL REFERENCES board_routes (id) ON DELETE CASCADE,
    grid_col        INT NOT NULL,
    grid_row        INT NOT NULL,
    position_x_cm   FLOAT,
    position_y_cm   FLOAT,
    hold_type       TEXT,
    role            TEXT NOT NULL CHECK (role IN ('start', 'hand', 'foot', 'finish')),
    board_angle_deg FLOAT,
    hand_sequence   INT
);
CREATE INDEX idx_route_holds_route_id ON route_holds (route_id);
CREATE INDEX idx_route_holds_role     ON route_holds (role);
CREATE INDEX idx_route_holds_grid     ON route_holds (grid_col, grid_row);

-- ────────────────────────────────────────────────────────────
-- SESSIONS
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    climber_id      UUID NOT NULL REFERENCES climbers (id) ON DELETE CASCADE,
    gym_id          UUID NOT NULL REFERENCES gyms (id) ON DELETE CASCADE,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at        TIMESTAMPTZ,
    board_angle_deg INT NOT NULL CHECK (board_angle_deg BETWEEN 0 AND 70),
    board_type      TEXT NOT NULL,
    video_path      TEXT,
    video_status    TEXT NOT NULL DEFAULT 'none'
        CHECK (video_status IN ('none', 'pending', 'processing', 'done', 'failed')),
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_sessions_climber_id   ON sessions (climber_id);
CREATE INDEX idx_sessions_gym_id       ON sessions (gym_id);
CREATE INDEX idx_sessions_started_at   ON sessions (started_at DESC);
CREATE INDEX idx_sessions_video_status ON sessions (video_status) WHERE video_status IN ('pending', 'processing');

-- ────────────────────────────────────────────────────────────
-- ATTEMPTS
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS attempts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      UUID NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    route_id        UUID NOT NULL REFERENCES board_routes (id),
    climber_id      UUID NOT NULL REFERENCES climbers (id),
    attempt_number  INT NOT NULL DEFAULT 1 CHECK (attempt_number >= 1),
    completed       BOOLEAN NOT NULL DEFAULT FALSE,
    duration_sec    FLOAT CHECK (duration_sec >= 0),
    video_offset_sec    FLOAT,
    video_end_sec       FLOAT,
    attempted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_attempts_session_id ON attempts (session_id);
CREATE INDEX idx_attempts_route_id   ON attempts (route_id);
CREATE INDEX idx_attempts_climber_id ON attempts (climber_id);
CREATE INDEX idx_attempts_completed  ON attempts (climber_id, completed);
CREATE UNIQUE INDEX idx_attempts_unique_try ON attempts (session_id, route_id, attempt_number);

-- ────────────────────────────────────────────────────────────
-- MOVE_EVENTS
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS move_events (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    attempt_id              UUID NOT NULL REFERENCES attempts (id) ON DELETE CASCADE,
    from_hold_id            UUID REFERENCES route_holds (id),
    to_hold_id              UUID NOT NULL REFERENCES route_holds (id),
    sequence_order          INT NOT NULL CHECK (sequence_order >= 1),
    move_type               TEXT,
    reach_distance_cm       FLOAT CHECK (reach_distance_cm >= 0),
    lateral_distance_cm     FLOAT,
    vertical_distance_cm    FLOAT,
    hand_used               TEXT CHECK (hand_used IN ('left', 'right', 'both', 'unknown')),
    timestamp_sec           FLOAT CHECK (timestamp_sec >= 0),
    succeeded               BOOLEAN
);
CREATE INDEX idx_move_events_attempt_id  ON move_events (attempt_id);
CREATE INDEX idx_move_events_move_type   ON move_events (move_type);
CREATE INDEX idx_move_events_to_hold     ON move_events (to_hold_id);
CREATE INDEX idx_move_events_sequence    ON move_events (attempt_id, sequence_order);

-- ────────────────────────────────────────────────────────────
-- POSE_FRAMES
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pose_frames (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    attempt_id              UUID NOT NULL REFERENCES attempts (id) ON DELETE CASCADE,
    move_event_id           UUID REFERENCES move_events (id),
    timestamp_sec           FLOAT NOT NULL CHECK (timestamp_sec >= 0),
    hip_angle_deg           FLOAT,
    shoulder_rot_deg        FLOAT,
    left_arm_reach_norm     FLOAT CHECK (left_arm_reach_norm BETWEEN 0.0 AND 1.5),
    right_arm_reach_norm    FLOAT CHECK (right_arm_reach_norm BETWEEN 0.0 AND 1.5),
    left_foot_angle_deg     FLOAT,
    right_foot_angle_deg    FLOAT,
    com_height_norm         FLOAT,
    tension_score           FLOAT CHECK (tension_score BETWEEN 0.0 AND 1.0),
    raw_landmarks           JSONB NOT NULL
);
CREATE INDEX idx_pose_frames_attempt_id    ON pose_frames (attempt_id);
CREATE INDEX idx_pose_frames_move_event_id ON pose_frames (move_event_id);
CREATE INDEX idx_pose_frames_timestamp     ON pose_frames (attempt_id, timestamp_sec);
CREATE INDEX idx_pose_frames_landmarks_gin ON pose_frames USING GIN (raw_landmarks);

-- ────────────────────────────────────────────────────────────
-- MOVE_LIBRARY
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS move_library (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name                    TEXT UNIQUE NOT NULL,
    category                TEXT NOT NULL,
    description             TEXT,
    required_body_positions TEXT[] NOT NULL DEFAULT '{}',
    primary_muscles         TEXT[] NOT NULL DEFAULT '{}',
    difficulty_tier         TEXT CHECK (difficulty_tier IN ('beginner','intermediate','advanced','elite')),
    grade_range_low         TEXT,
    grade_range_high        TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_move_library_category ON move_library (category);
CREATE INDEX idx_move_library_tier     ON move_library (difficulty_tier);

-- ────────────────────────────────────────────────────────────
-- CLIMBER_MOVE_STATS
-- ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS climber_move_stats (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    climber_id              UUID NOT NULL REFERENCES climbers (id) ON DELETE CASCADE,
    move_library_id         UUID NOT NULL REFERENCES move_library (id),
    total_attempts          INT NOT NULL DEFAULT 0,
    successful_attempts     INT NOT NULL DEFAULT 0,
    success_rate            FLOAT GENERATED ALWAYS AS (
        CASE WHEN total_attempts = 0 THEN NULL
             ELSE successful_attempts::FLOAT / total_attempts
        END
    ) STORED,
    vs_peers_percentile     INT CHECK (vs_peers_percentile BETWEEN 0 AND 100),
    exposure_ratio          FLOAT CHECK (exposure_ratio >= 0),
    is_flagged_weakness     BOOLEAN NOT NULL DEFAULT FALSE,
    last_updated            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX idx_climber_move_stats_unique    ON climber_move_stats (climber_id, move_library_id);
CREATE INDEX idx_climber_move_stats_climber          ON climber_move_stats (climber_id);
CREATE INDEX idx_climber_move_stats_weakness         ON climber_move_stats (climber_id, is_flagged_weakness) WHERE is_flagged_weakness = TRUE;
CREATE INDEX idx_climber_move_stats_percentile       ON climber_move_stats (vs_peers_percentile);

-- ────────────────────────────────────────────────────────────
-- TRIGGERS — auto-update updated_at
-- ────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER gyms_updated_at     BEFORE UPDATE ON gyms     FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
CREATE TRIGGER climbers_updated_at BEFORE UPDATE ON climbers FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
