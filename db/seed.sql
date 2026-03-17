-- seed.sql — Development seed data
-- Populates the database with realistic test data.
-- Run after schema.sql:
--   psql climbing_platform < db/seed.sql

-- ── Gyms ──────────────────────────────────────────────────
INSERT INTO gyms (id, name, location, kilter_board_angle, has_kilter_board, has_moon_board, timezone)
VALUES
    ('00000000-0000-0000-0000-000000000001', 'Chicago Climbing Co.',    'Chicago, IL',    40, TRUE, TRUE,  'America/Chicago'),
    ('00000000-0000-0000-0000-000000000002', 'Bloc Shop Denver',        'Denver, CO',     45, TRUE, FALSE, 'America/Denver'),
    ('00000000-0000-0000-0000-000000000003', 'Movement Glendale',       'Glendale, CA',   40, TRUE, TRUE,  'America/Los_Angeles');

-- ── Climbers ──────────────────────────────────────────────
INSERT INTO climbers (id, gym_id, name, email, self_reported_grade, goals, weak_move_types, years_climbing, height_cm, wingspan_cm, video_consent)
VALUES
    ('00000000-0000-0000-0001-000000000001',
     '00000000-0000-0000-0000-000000000001',
     'Jordan Kim', 'jordan@example.com', 'V6',
     ARRAY['climb V8 by end of year', 'improve slopers'],
     ARRAY['sloper', 'compression'], 4, 175, 178, TRUE),

    ('00000000-0000-0000-0001-000000000002',
     '00000000-0000-0000-0000-000000000001',
     'Alex Torres', 'alex@example.com', 'V4',
     ARRAY['get stronger on overhangs', 'learn heel hooks'],
     ARRAY['heel_hook', 'dyno'], 2, 168, 165, TRUE),

    ('00000000-0000-0000-0001-000000000003',
     '00000000-0000-0000-0000-000000000001',
     'Sam Rivera', 'sam@example.com', 'V9',
     ARRAY['project V11', 'fix footwork'],
     ARRAY['toe_hook'], 8, 180, 185, FALSE);

-- ── Sample board routes (Kilter) ──────────────────────────
INSERT INTO board_routes (id, source, external_id, board_type, board_angle_deg, community_grade, difficulty_score, send_count)
VALUES
    ('00000000-0000-0000-0002-000000000001', 'kilter', 'k-001', 'kilter_original', 40, 'V4', 0.35, 1240),
    ('00000000-0000-0000-0002-000000000002', 'kilter', 'k-002', 'kilter_original', 40, 'V6', 0.52, 876),
    ('00000000-0000-0000-0002-000000000003', 'kilter', 'k-003', 'kilter_original', 40, 'V8', 0.68, 412),
    ('00000000-0000-0000-0002-000000000004', 'moonboard', 'mb-001', 'moonboard_2024', 40, 'V5', 0.43, 2100),
    ('00000000-0000-0000-0002-000000000005', 'moonboard', 'mb-002', 'moonboard_2024', 40, 'V7', 0.60, 980);

-- ── Route holds (sample for route k-001) ──────────────────
INSERT INTO route_holds (route_id, grid_col, grid_row, role, hold_type, hand_sequence, position_x_cm, position_y_cm)
VALUES
    ('00000000-0000-0000-0002-000000000001', 4,  2,  'start',  'jug',    1,  56.0,  28.0),
    ('00000000-0000-0000-0002-000000000001', 7,  2,  'start',  'jug',    2,  98.0,  28.0),
    ('00000000-0000-0000-0002-000000000001', 5,  6,  'hand',   'crimp',  3,  70.0,  84.0),
    ('00000000-0000-0000-0002-000000000001', 9,  9,  'hand',   'sloper', 4, 126.0, 126.0),
    ('00000000-0000-0000-0002-000000000001', 3,  4,  'foot',   'foot_chip', NULL, 42.0, 56.0),
    ('00000000-0000-0000-0002-000000000001', 7,  7,  'foot',   'foot_chip', NULL, 98.0, 98.0),
    ('00000000-0000-0000-0002-000000000001', 7,  13, 'finish', 'jug',    5,  98.0, 182.0);
