-- Migration 004: move events and pose frames
-- This is the most granular data layer.
-- move_events breaks an attempt into individual transitions between holds.
-- pose_frames stores the body position captured for each move via MediaPipe.

-- ─────────────────────────────────────────
-- MOVE_EVENTS
-- One row per hold-to-hold transition
-- within an attempt.
-- from_hold_id = NULL means the climber
-- started from the ground (first move).
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS move_events (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    attempt_id          UUID NOT NULL REFERENCES attempts (id) ON DELETE CASCADE,

    -- The two holds involved in this transition.
    -- Both reference route_holds rows on the same route.
    from_hold_id        UUID REFERENCES route_holds (id),   -- NULL on the first move
    to_hold_id          UUID NOT NULL REFERENCES route_holds (id),

    -- Position in the move sequence for this attempt (1-indexed)
    sequence_order      INT NOT NULL CHECK (sequence_order >= 1),

    -- Classified move type — matches move_library.name
    -- Set by the ML classifier after pose processing.
    -- NULL until classified.
    move_type           TEXT,

    -- Straight-line distance between hold centres (cm), computed from grid coords
    reach_distance_cm   FLOAT CHECK (reach_distance_cm >= 0),

    -- Horizontal (lateral) component of the reach
    lateral_distance_cm FLOAT,

    -- Vertical component of the reach
    vertical_distance_cm FLOAT,

    -- Which hand made the move: 'left' | 'right' | 'both' | 'unknown'
    hand_used           TEXT CHECK (hand_used IN ('left', 'right', 'both', 'unknown')),

    -- Timestamp within the attempt video when this move started
    timestamp_sec       FLOAT CHECK (timestamp_sec >= 0),

    -- Whether the move succeeded (climber stuck the hold)
    -- NULL = not yet determined
    succeeded           BOOLEAN
);

CREATE INDEX idx_move_events_attempt_id     ON move_events (attempt_id);
CREATE INDEX idx_move_events_move_type      ON move_events (move_type);
CREATE INDEX idx_move_events_to_hold        ON move_events (to_hold_id);
CREATE INDEX idx_move_events_sequence       ON move_events (attempt_id, sequence_order);


-- ─────────────────────────────────────────
-- POSE_FRAMES
-- One row per MediaPipe sample captured
-- during an attempt.
-- Typically ~10 frames per move event
-- (subsampled from 30fps raw video).
--
-- Angles are in degrees.
-- Distances are normalised to torso length
-- (so they're comparable across climbers).
-- raw_landmarks stores the full 33-point
-- MediaPipe output as JSONB for reprocessing.
-- ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pose_frames (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    attempt_id          UUID NOT NULL REFERENCES attempts (id) ON DELETE CASCADE,

    -- Link to the specific move this frame belongs to.
    -- NULL if frame hasn't been assigned to a move yet.
    move_event_id       UUID REFERENCES move_events (id),

    -- Timestamp within the attempt video
    timestamp_sec       FLOAT NOT NULL CHECK (timestamp_sec >= 0),

    -- ── Core body position metrics ───────
    -- Hip angle: angle of the hip joint (lower = more open hip, higher = closed/flagged)
    hip_angle_deg           FLOAT,

    -- Shoulder rotation: how much the shoulders are rotated from square-on
    shoulder_rot_deg        FLOAT,

    -- Arm reach distances, normalised to climber torso length
    -- 1.0 = fully extended, 0.0 = arm folded in
    left_arm_reach_norm     FLOAT CHECK (left_arm_reach_norm BETWEEN 0.0 AND 1.5),
    right_arm_reach_norm    FLOAT CHECK (right_arm_reach_norm BETWEEN 0.0 AND 1.5),

    -- Foot position — angle of the foot relative to the wall
    -- Positive = flagged outward, Negative = flagged inward
    left_foot_angle_deg     FLOAT,
    right_foot_angle_deg    FLOAT,

    -- Centre of mass height, normalised to climber height
    -- Useful for detecting drop-knees, heel hooks, high steps
    com_height_norm         FLOAT,

    -- Body tension score — how aligned core/hips are (0.0–1.0)
    -- Derived metric, computed in pose_processor.py
    tension_score           FLOAT CHECK (tension_score BETWEEN 0.0 AND 1.0),

    -- ── Raw data ─────────────────────────
    -- Full 33-landmark output from MediaPipe Pose.
    -- Schema: {"landmarks": [{"x": 0.5, "y": 0.3, "z": 0.1, "visibility": 0.99}, ...]}
    -- Kept for reprocessing when models improve.
    raw_landmarks           JSONB NOT NULL
);

CREATE INDEX idx_pose_frames_attempt_id     ON pose_frames (attempt_id);
CREATE INDEX idx_pose_frames_move_event_id  ON pose_frames (move_event_id);
CREATE INDEX idx_pose_frames_timestamp      ON pose_frames (attempt_id, timestamp_sec);

-- GIN index on JSONB for landmark queries
CREATE INDEX idx_pose_frames_landmarks_gin  ON pose_frames USING GIN (raw_landmarks);
