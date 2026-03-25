#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# db/migrate_supabase.sh
# Run all migrations against Supabase in order.
#
# Usage:
#   DB_PASSWORD=<your-supabase-db-password> bash db/migrate_supabase.sh
#
# Or export it first:
#   export DB_PASSWORD=yourpassword
#   bash db/migrate_supabase.sh
# ──────────────────────────────────────────────────────────────────────────────

set -e

PROJECT_REF="pycvlovkepsargcnvyqc"
DB_USER="postgres"
DB_HOST="db.${PROJECT_REF}.supabase.co"
DB_PORT="5432"
DB_NAME="postgres"

if [ -z "$DB_PASSWORD" ]; then
  echo "Error: DB_PASSWORD is not set."
  echo "Run:  DB_PASSWORD=yourpassword bash db/migrate_supabase.sh"
  exit 1
fi

CONN="postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

MIGRATIONS=(
  "db/migrations/001_gyms_climbers.sql"
  "db/migrations/002_board_routes.sql"
  "db/migrations/003_sessions_attempts.sql"
  "db/migrations/004_move_events_pose.sql"
  "db/migrations/005_move_library_stats.sql"
  "db/migrations/006_climber_profiles_pdscore.sql"
  "db/migrations/007_video_upload_coaching.sql"
)

echo "→ Connecting to Supabase project: ${PROJECT_REF}"
echo ""

for f in "${MIGRATIONS[@]}"; do
  echo "  Running $f …"
  psql "$CONN" -f "$f" -v ON_ERROR_STOP=1 --quiet
  echo "  ✓ done"
done

echo ""
echo "All migrations applied to Supabase."
