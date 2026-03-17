# Climbing Intelligence Platform — Database Layer

## Project structure

```
climbing-platform/
├── db/
│   ├── migrations/          # Numbered SQL migration files — run in order
│   │   ├── 001_gyms_climbers.sql
│   │   ├── 002_board_routes.sql
│   │   ├── 003_sessions_attempts.sql
│   │   ├── 004_move_events_pose.sql
│   │   └── 005_move_library_stats.sql
│   ├── models/              # Python dataclasses mirroring each table
│   │   ├── __init__.py
│   │   ├── gym.py
│   │   ├── climber.py
│   │   ├── route.py
│   │   ├── session.py
│   │   └── move.py
│   ├── schema.sql           # Full schema in one file (generated from migrations)
│   └── seed.sql             # Dev seed data for testing
├── pipeline/
│   ├── kilter_sync.py       # Pulls routes from Kilter Board API
│   ├── moonboard_sync.py    # Imports Moon Board dataset
│   └── pose_processor.py   # Processes MediaPipe output into pose_frames
├── tests/
│   └── test_models.py
├── requirements.txt
└── README.md
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create a PostgreSQL database
createdb climbing_platform

# Run migrations in order
psql climbing_platform < db/migrations/001_gyms_climbers.sql
psql climbing_platform < db/migrations/002_board_routes.sql
psql climbing_platform < db/migrations/003_sessions_attempts.sql
psql climbing_platform < db/migrations/004_move_events_pose.sql
psql climbing_platform < db/migrations/005_move_library_stats.sql

# Or run the full schema at once
psql climbing_platform < db/schema.sql

# Load dev seed data
psql climbing_platform < db/seed.sql
```

## Design decisions

- **PostgreSQL** — chosen for native UUID, JSONB (pose landmarks), array columns (goals, weak_move_types), and strong indexing on spatial queries.
- **Migrations over ORM** — raw SQL migrations are version-controlled and environment-agnostic. No magic, no lock-in. Each migration file is numbered and idempotent.
- **JSONB for raw_landmarks** — MediaPipe returns 33 landmarks per frame. Storing as JSONB lets you reprocess with updated models without a schema change.
- **Computed stats table** — `climber_move_stats` is a pre-aggregated table, not a view, for fast dashboard queries. It gets refreshed by a background job after each session.
- **board_type + source fields** — designed to support Kilter, Moon Board, and future custom boards without schema changes.
