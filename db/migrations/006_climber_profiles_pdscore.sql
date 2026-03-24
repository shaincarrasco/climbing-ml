-- ──────────────────────────────────────────────────────────────────────────────
-- Migration 006: Climber profiles + Personal Difficulty Score (PDScore)
--
-- Purpose:
--   Introduce a lightweight climber identity layer that enables personalised
--   grade predictions. Accounts are optional — all profile data can be provided
--   anonymously via localStorage. The schema is auth-ready (add password_hash
--   and an auth middleware when you want to enforce login).
--
-- Tables introduced:
--   climber_profiles         Physical + skill attributes of a climber
--   climber_move_affinity    Per-move-type success rates (updated after each send)
--   route_sends              Personal send log (links climber → route)
--
-- PDScore formula (computed at query time in /api/predict):
--   base  = XGBoost difficulty_score for the route
--   body  = Σ(route_feature_delta × body_weight)   — height, wingspan, weight
--   moves = Σ(move_type_weight × (1 - success_rate)) — per-type skill gap
--   pdscore = clamp(base + body + moves, 0, 1)
-- ──────────────────────────────────────────────────────────────────────────────

-- ── Climber profiles ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS climber_profiles (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identity (optional — null until user sets it)
    display_name    TEXT,
    email           TEXT UNIQUE,                    -- null for anonymous profiles

    -- Body measurements (used for PDScore body modifier)
    height_cm       FLOAT,
    wingspan_cm     FLOAT,
    weight_kg       FLOAT,
    shoe_size_eu    FLOAT,                          -- proxy for foot/reach ratio

    -- Computed on save (stored for query performance)
    ape_index_cm    FLOAT GENERATED ALWAYS AS (wingspan_cm - height_cm) STORED,
    bmi             FLOAT GENERATED ALWAYS AS (weight_kg / NULLIF((height_cm / 100.0)^2, 0)) STORED,

    -- Climbing ability
    onsight_grade   TEXT,                           -- e.g. 'V6'
    experience_yrs  FLOAT,                          -- 0.5 = 6 months, 5 = five years

    -- Preferences
    preferred_angle INT,                            -- 0–70°
    preferred_style TEXT,                           -- 'dynamic'|'static'|'balance'|'power'

    -- Session tracking
    last_active_at  TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW()
);

COMMENT ON TABLE climber_profiles IS
    'One row per climber. id is stored in browser localStorage for anonymous use.';

COMMENT ON COLUMN climber_profiles.ape_index_cm IS
    'wingspan - height. Positive = reach advantage (affects big-move difficulty).';


-- ── Per-move-type affinity ────────────────────────────────────────────────────
-- Tracks success rates on each route style to personalise the difficulty score.
-- Updated after each logged send; also seeded from onsight_grade on first use.

CREATE TABLE IF NOT EXISTS climber_move_affinity (
    climber_id          UUID    NOT NULL REFERENCES climber_profiles(id) ON DELETE CASCADE,
    move_type           TEXT    NOT NULL,   -- see MOVE_TYPES below

    -- Raw stats
    attempts            INT     NOT NULL DEFAULT 0,
    sends               INT     NOT NULL DEFAULT 0,
    avg_attempts_to_send FLOAT,             -- NULL until first send

    -- Derived: success_rate = sends / NULLIF(attempts, 0)
    -- Stored for query perf; refreshed on every update
    success_rate        FLOAT   GENERATED ALWAYS AS (
                            sends::float / NULLIF(attempts, 0)
                        ) STORED,

    last_updated        TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (climber_id, move_type)
);

-- Valid move_type values (enforced by application layer):
--   dyno          — big explosive moves (dyno_flag=1)
--   high_tension  — routes with avg_tension > 0.65
--   crimp         — crimp_ratio > 0.40
--   sloper        — sloper_ratio > 0.30
--   pinch         — pinch_ratio > 0.25
--   lateral       — lateral_ratio > 0.50
--   sustained     — moves_count > 9
--   overhang_40   — board_angle between 35-44°
--   overhang_50   — board_angle between 45-54°
--   overhang_60   — board_angle ≥ 55°
--   compression   — sloper + high vertical ratio
--   coordination  — high zigzag_ratio (> 0.30)

COMMENT ON TABLE climber_move_affinity IS
    'Per-climber success rates on each climbing style. Drives the move modifier in PDScore.';


-- ── Personal send log ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS route_sends (
    id              SERIAL      PRIMARY KEY,
    climber_id      UUID        NOT NULL REFERENCES climber_profiles(id) ON DELETE CASCADE,
    route_id        UUID        NOT NULL REFERENCES board_routes(id),

    -- Send details
    attempts        INT         NOT NULL DEFAULT 1,
    sent_at         TIMESTAMP   DEFAULT NOW(),
    board_angle_deg INT,                            -- angle attempted (may differ from route default)
    notes           TEXT,                           -- 'skipped the match', 'used kneebar', etc.

    -- Auto-populated from route on insert
    route_grade     TEXT,                           -- snapshot; grade may change
    route_difficulty_score FLOAT,

    CONSTRAINT sends_once_per_day UNIQUE (climber_id, route_id)
);

COMMENT ON TABLE route_sends IS
    'User send log. Powers move affinity updates and strength/weakness analysis.';


-- ── Indexes ───────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_sends_climber     ON route_sends (climber_id);
CREATE INDEX IF NOT EXISTS idx_sends_route       ON route_sends (route_id);
CREATE INDEX IF NOT EXISTS idx_affinity_climber  ON climber_move_affinity (climber_id);
CREATE INDEX IF NOT EXISTS idx_profiles_email    ON climber_profiles (email) WHERE email IS NOT NULL;
