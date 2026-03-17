-- Migration 005: move library and climber move stats
-- move_library is the taxonomy of climbing move types.
-- climber_move_stats is a pre-aggregated gap-analysis table
-- that powers the dashboard and recommendation engine.

-- ─────────────────────────────────────────
-- MOVE_LIBRARY
-- The canonical list of move types.
-- Seeded once; updated manually as new
-- move categories are identified.
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS move_library (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Short identifier used in move_events.move_type and climbers.weak_move_types
    -- e.g. 'crimp', 'sloper', 'heel_hook', 'dyno', 'compression'
    name                    TEXT UNIQUE NOT NULL,

    -- High-level grouping for dashboard display
    -- 'grip_strength' | 'body_position' | 'dynamic' | 'footwork' | 'technique'
    category                TEXT NOT NULL,

    description             TEXT,

    -- Body positions typically required for this move type.
    -- Array of descriptive strings, e.g.:
    -- '{"open hip", "high step", "shoulder rotation > 30deg"}'
    required_body_positions TEXT[] NOT NULL DEFAULT '{}',

    -- Which muscle groups this move primarily trains
    -- '{"finger flexors", "core", "lat pull"}'
    primary_muscles         TEXT[] NOT NULL DEFAULT '{}',

    -- Broad difficulty tier for this move type across all grades
    -- 'beginner' | 'intermediate' | 'advanced' | 'elite'
    difficulty_tier         TEXT CHECK (difficulty_tier IN ('beginner','intermediate','advanced','elite')),

    -- Typical grade range where this move first appears and dominates
    grade_range_low         TEXT,   -- e.g. 'V0'
    grade_range_high        TEXT,   -- e.g. 'V17'

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_move_library_category ON move_library (category);
CREATE INDEX idx_move_library_tier     ON move_library (difficulty_tier);


-- ─────────────────────────────────────────
-- CLIMBER_MOVE_STATS
-- Pre-aggregated stats per climber per move
-- type. This is the core gap-detection table.
--
-- Refreshed by a background job after each
-- session completes video processing.
-- NOT calculated on the fly — too slow for
-- live dashboard queries.
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS climber_move_stats (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    climber_id              UUID NOT NULL REFERENCES climbers (id) ON DELETE CASCADE,
    move_library_id         UUID NOT NULL REFERENCES move_library (id),

    -- Raw counts
    total_attempts          INT NOT NULL DEFAULT 0,
    successful_attempts     INT NOT NULL DEFAULT 0,

    -- Computed success rate (0.0–1.0)
    success_rate            FLOAT GENERATED ALWAYS AS (
        CASE WHEN total_attempts = 0 THEN NULL
             ELSE successful_attempts::FLOAT / total_attempts
        END
    ) STORED,

    -- How this climber compares to peers at the same grade level.
    -- Stored as a percentile string for readability: '23rd', '87th'
    -- Recomputed nightly via a background job.
    vs_peers_percentile     INT CHECK (vs_peers_percentile BETWEEN 0 AND 100),

    -- Exposure score: how often this move type appears in the routes
    -- they've attempted vs. how often it appears in routes at their grade.
    -- < 1.0 = under-exposed (potential gap), > 1.0 = over-indexed
    exposure_ratio          FLOAT CHECK (exposure_ratio >= 0),

    -- Flag set by the gap detector when this is an identified weakness
    is_flagged_weakness     BOOLEAN NOT NULL DEFAULT FALSE,

    last_updated            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- One row per climber per move type — no duplicates
CREATE UNIQUE INDEX idx_climber_move_stats_unique
    ON climber_move_stats (climber_id, move_library_id);

CREATE INDEX idx_climber_move_stats_climber     ON climber_move_stats (climber_id);
CREATE INDEX idx_climber_move_stats_weakness    ON climber_move_stats (climber_id, is_flagged_weakness)
    WHERE is_flagged_weakness = TRUE;
CREATE INDEX idx_climber_move_stats_percentile  ON climber_move_stats (vs_peers_percentile);


-- ─────────────────────────────────────────
-- SEED: move_library initial data
-- Add or modify move types here as the
-- classification system evolves.
-- ─────────────────────────────────────────
INSERT INTO move_library (name, category, description, required_body_positions, primary_muscles, difficulty_tier, grade_range_low, grade_range_high)
VALUES
    ('crimp',           'grip_strength',  'Small edge gripped with curled fingers',
     ARRAY['closed hand position', 'shoulder engagement'],
     ARRAY['finger flexors', 'forearm'], 'beginner', 'V0', 'V17'),

    ('open_hand',       'grip_strength',  'Large hold gripped with open fingers',
     ARRAY['relaxed grip', 'low shoulder'],
     ARRAY['finger flexors', 'forearm'], 'beginner', 'V0', 'V17'),

    ('sloper',          'grip_strength',  'Rounded hold requiring friction and body position',
     ARRAY['low hips', 'arms straight', 'shoulders over hold'],
     ARRAY['shoulder stabilisers', 'core', 'forearm'], 'intermediate', 'V3', 'V17'),

    ('pinch',           'grip_strength',  'Hold squeezed between thumb and fingers',
     ARRAY['shoulder engagement', 'elbow at 90deg'],
     ARRAY['thumb adductors', 'forearm', 'shoulder'], 'intermediate', 'V2', 'V17'),

    ('pocket',          'grip_strength',  'Small hole for one or two fingers',
     ARRAY['precise finger placement', 'controlled weight transfer'],
     ARRAY['finger flexors', 'A2 pulley'], 'advanced', 'V4', 'V17'),

    ('compression',     'body_position',  'Two-handed squeeze on a feature or volume',
     ARRAY['hip into wall', 'shoulder squeeze', 'core tension'],
     ARRAY['chest', 'biceps', 'core', 'adductors'], 'intermediate', 'V3', 'V17'),

    ('heel_hook',       'footwork',       'Heel placed on a hold to generate upward force',
     ARRAY['hip open', 'core tension', 'heel engagement'],
     ARRAY['hamstrings', 'glutes', 'core'], 'intermediate', 'V3', 'V17'),

    ('toe_hook',        'footwork',       'Top of foot hooked on a hold',
     ARRAY['hip rotation', 'quad engagement'],
     ARRAY['hip flexors', 'quads', 'core'], 'intermediate', 'V4', 'V17'),

    ('drop_knee',       'body_position',  'Inside knee twisted down to open hip',
     ARRAY['hip open', 'knee drop', 'shoulder over hold'],
     ARRAY['hip rotators', 'quads', 'core'], 'intermediate', 'V3', 'V17'),

    ('flag',            'body_position',  'One leg extended sideways for balance',
     ARRAY['hip open', 'shoulder rotation', 'one-leg balance'],
     ARRAY['hip abductors', 'core', 'calf'], 'beginner', 'V1', 'V17'),

    ('dyno',            'dynamic',        'Dynamic move leaving the wall entirely',
     ARRAY['explosive hip drive', 'arm swing', 'full body tension'],
     ARRAY['hip flexors', 'lats', 'explosive quads'], 'advanced', 'V4', 'V17'),

    ('deadpoint',       'dynamic',        'Controlled dynamic move catching at peak height',
     ARRAY['hip drive', 'precise timing', 'controlled swing'],
     ARRAY['lats', 'core', 'quads'], 'intermediate', 'V3', 'V17'),

    ('undercling',      'technique',      'Hold pulled upward with palms facing up',
     ARRAY['hips low', 'lean back', 'core engagement'],
     ARRAY['biceps', 'core', 'hip flexors'], 'intermediate', 'V2', 'V17'),

    ('gaston',          'technique',      'Hold pushed outward with elbow pointing out',
     ARRAY['elbow flared', 'shoulder external rotation'],
     ARRAY['shoulder rotators', 'chest', 'forearm'], 'advanced', 'V5', 'V17'),

    ('mantle',          'technique',      'Pressing down on a hold to top out',
     ARRAY['tricep press', 'shoulder over hold', 'wrist rotation'],
     ARRAY['triceps', 'shoulder stabilisers'], 'intermediate', 'V2', 'V17')

ON CONFLICT (name) DO NOTHING;
