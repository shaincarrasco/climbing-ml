-- Migration 001: gyms and climbers
-- Run first — all other tables depend on these two.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────────
-- GYMS
-- One row per physical gym location.
-- A gym can have multiple board types at
-- different angles (tracked in sessions).
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS gyms (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name                TEXT NOT NULL,
    location            TEXT,
    -- Default board angle for this gym's Kilter Board (degrees from vertical)
    -- Kilter standard angles: 0, 20, 35, 40, 45, 50, 55, 60, 65, 70
    kilter_board_angle  INT CHECK (kilter_board_angle BETWEEN 0 AND 70),
    has_moon_board      BOOLEAN NOT NULL DEFAULT FALSE,
    has_kilter_board    BOOLEAN NOT NULL DEFAULT FALSE,
    -- Contact info for multi-gym deployments
    contact_email       TEXT,
    timezone            TEXT NOT NULL DEFAULT 'America/Chicago',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_gyms_name ON gyms (name);


-- ─────────────────────────────────────────
-- CLIMBERS
-- One row per gym member.
-- Self-reported fields are filled at signup
-- and updated as the climber progresses.
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS climbers (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    gym_id                  UUID NOT NULL REFERENCES gyms (id) ON DELETE CASCADE,
    name                    TEXT NOT NULL,
    email                   TEXT UNIQUE,

    -- Self-reported grade: V-scale for bouldering (V0–V17)
    -- or French sport grade (5a–9c) — stored as text to handle both
    self_reported_grade     TEXT,

    -- Array of free-text or structured goals, e.g.
    -- '{"get better at slopers", "climb V7", "improve footwork"}'
    goals                   TEXT[] NOT NULL DEFAULT '{}',

    -- Move types the climber knows they struggle with.
    -- Seeded at signup, updated by the system over time.
    -- Values match move_library.name
    weak_move_types         TEXT[] NOT NULL DEFAULT '{}',

    years_climbing          INT CHECK (years_climbing >= 0),
    height_cm               INT,   -- useful for normalizing reach distances
    wingspan_cm             INT,   -- ape index data point

    -- Whether this climber has consented to video capture
    video_consent           BOOLEAN NOT NULL DEFAULT FALSE,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_climbers_gym_id ON climbers (gym_id);
CREATE INDEX idx_climbers_grade  ON climbers (self_reported_grade);


-- ─────────────────────────────────────────
-- Auto-update updated_at on any row change
-- ─────────────────────────────────────────
CREATE OR REPLACE FUNCTION touch_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER gyms_updated_at
    BEFORE UPDATE ON gyms
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();

CREATE TRIGGER climbers_updated_at
    BEFORE UPDATE ON climbers
    FOR EACH ROW EXECUTE FUNCTION touch_updated_at();
