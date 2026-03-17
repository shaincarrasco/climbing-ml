-- Migration 002: board routes and holds
-- Stores every route imported from Kilter / Moon Board APIs.
-- board_routes is the ML training backbone — community grades
-- are the prediction target, hold positions are the features.

-- ─────────────────────────────────────────
-- BOARD_ROUTES
-- One row per unique route from any board.
-- source = 'kilter' | 'moonboard' | 'custom'
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS board_routes (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Where this route came from
    source              TEXT NOT NULL CHECK (source IN ('kilter', 'moonboard', 'custom')),
    -- The route's ID in the external system, for deduplication on re-sync
    external_id         TEXT,

    board_type          TEXT NOT NULL,   -- 'kilter_original', 'kilter_home', 'moonboard_2024', etc.
    board_angle_deg     INT NOT NULL CHECK (board_angle_deg BETWEEN 0 AND 70),

    name                TEXT,            -- optional setter-given name
    setter_name         TEXT,

    -- Grade as stored by the source platform
    -- e.g. 'V6', '7A', '6C+'
    community_grade     TEXT,

    -- Normalised numeric difficulty (0.0–1.0) derived from community grade.
    -- Pre-computed for ML feature use.
    difficulty_score    FLOAT CHECK (difficulty_score BETWEEN 0.0 AND 1.0),

    -- How many people have logged a send in the source platform
    send_count          INT NOT NULL DEFAULT 0,

    -- Average star rating from the source platform (1–3 for Kilter)
    avg_quality_rating  FLOAT,

    -- Source-platform metadata as flexible JSON
    -- e.g. {"frames_count": 4, "is_benchmark": true}
    metadata            JSONB NOT NULL DEFAULT '{}',

    -- When we last pulled this from the API
    synced_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Prevent duplicate imports on re-sync
CREATE UNIQUE INDEX idx_board_routes_external
    ON board_routes (source, external_id)
    WHERE external_id IS NOT NULL;

CREATE INDEX idx_board_routes_source        ON board_routes (source);
CREATE INDEX idx_board_routes_grade         ON board_routes (community_grade);
CREATE INDEX idx_board_routes_difficulty    ON board_routes (difficulty_score);
CREATE INDEX idx_board_routes_send_count    ON board_routes (send_count DESC);


-- ─────────────────────────────────────────
-- ROUTE_HOLDS
-- Every hold placed on a route.
-- Grid coordinates follow each board's
-- native coordinate system:
--   Kilter:    col 0–17, row 0–17 (0,0 = bottom-left)
--   Moon Board: col 0–10, row 0–17
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS route_holds (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    route_id        UUID NOT NULL REFERENCES board_routes (id) ON DELETE CASCADE,

    -- Native board grid position
    grid_col        INT NOT NULL,
    grid_row        INT NOT NULL,

    -- Physical position in cm from board origin (computed from grid + angle)
    -- Pre-computed to avoid repeated trig in the ML pipeline
    position_x_cm   FLOAT,
    position_y_cm   FLOAT,

    -- hold_type describes the physical hold shape on the board.
    -- For Kilter: maps to hold IDs from the API.
    -- Values: 'crimp', 'sloper', 'pinch', 'jug', 'pocket', 'volume', 'foot_chip'
    hold_type       TEXT,

    -- Role of this hold within the route
    role            TEXT NOT NULL CHECK (role IN ('start', 'hand', 'foot', 'finish')),

    -- The board angle this hold was set for
    -- (a hold can be at a different effective angle than the board default)
    board_angle_deg FLOAT,

    -- Sequence order among hand holds (NULL for foot-only holds)
    -- Used to define the intended move sequence
    hand_sequence   INT
);

CREATE INDEX idx_route_holds_route_id   ON route_holds (route_id);
CREATE INDEX idx_route_holds_role       ON route_holds (role);
CREATE INDEX idx_route_holds_grid       ON route_holds (grid_col, grid_row);
