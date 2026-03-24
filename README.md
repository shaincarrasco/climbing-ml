# Climbing ML Platform

A full-stack machine learning platform for predicting climbing route difficulty, analyzing climber movement, and visualizing beta on the Kilter Board.

## Overview

The system combines route geometry (hold positions, angles, types) with human pose data extracted from real climbing videos to train an XGBoost difficulty model across ~57,000 Kilter Board routes.

**Core insight:** Most routes have no video. A two-stage imputation approach predicts pose metrics from hold geometry alone, giving every route a pose profile — not just the ~2% with real video coverage.

### Architecture

```
Kilter Board API                 Climbing Videos (Instagram / YouTube)
      │                                         │
kilter_sync.py                    youtube_scraper.py / beta_scraper.py
      │                                         │
PostgreSQL (board_routes,              pose_extractor.py
 holds, sessions, attempts)           (MediaPipe → pose_frames)
      │                                         │
feature_extraction.py            pose_feature_extraction.py
      │                                         │
routes_features.csv              pose_features.csv
      │                                         │
      └──────────────┬──────────────────────────┘
                     │
             pose_imputer.py          ← Stage 1: geometry → pose (covers all 57K routes)
                     │
             pose_imputed.csv
                     │
             hold_type_classifier.py  ← enriches hold-type ratios for unlabeled holds
                     │
             difficulty_model.py      ← Stage 2: main XGBoost grade predictor
             + quantile regressors    ← prediction intervals (q10 / q90)
                     │
              Flask API (api/app.py)
                     │
              Web UI (index.html)
```

## Model Performance

| Model | Metric | Value |
|-------|--------|-------|
| Main difficulty | Within 1 grade (V-scale) | **83.3%** |
| Main difficulty | Within 2 grades | ~94% |
| Pose imputer | Median CV MAE | < 0.03 (normalized) |
| Hold type classifier | Stratified CV accuracy | ~72% |

## Project Structure

```
climbing-ml/
├── api/
│   └── app.py                    # Flask REST API with prediction caching
├── db/
│   ├── migrations/               # Numbered SQL migrations (run in order)
│   ├── models/                   # Python dataclasses mirroring DB tables
│   ├── schema.sql                # Full schema (generated from migrations)
│   └── seed.sql                  # Dev seed data
├── ml/
│   ├── difficulty_model.py       # Main XGBoost grade predictor (+ quantile models)
│   ├── pose_difficulty_model.py  # Pose-only grade model (video routes only)
│   ├── pose_imputer.py           # Stage 1: hold geometry → pose metrics
│   ├── hold_type_classifier.py   # Multiclass hold type prediction (jug/crimp/sloper/pinch)
│   ├── evaluate.py               # Offline evaluation + grade boundary calibration
│   └── feature_importance.py     # SHAP / feature importance analysis
├── pipeline/
│   ├── kilter_sync.py            # Kilter Board API → PostgreSQL sync
│   ├── feature_extraction.py     # Routes → geometry feature CSV
│   ├── pose_extractor.py         # Video → MediaPipe → pose_frames (PostgreSQL)
│   ├── pose_feature_extraction.py# pose_frames → per-route pose CSV
│   ├── beta_scraper.py           # Bulk Instagram/YouTube beta download + pose extraction
│   ├── youtube_scraper.py        # YouTube-specific scraper
│   ├── save_correlations.py      # Pearson correlations: pose ↔ grade
│   └── data_quality.py           # Quality scoring + bad-video purge
├── js/
│   ├── app.js                    # Route explorer / search
│   ├── creator.js                # Route creator with animated climber
│   ├── board.js                  # Hold rendering (Kilter layout)
│   ├── explore.js                # Grade distribution explorer
│   ├── profile.js                # Climber profile + progress tracking
│   └── stick-figure.js           # SVG stick figure animator
├── css/
│   └── styles.css
├── tests/
│   └── test_models.py
├── index.html                    # Single-page app entry point
├── retrain.py                    # Orchestrates full model retrain pipeline
└── requirements.txt
```

## Setup

### Prerequisites
- Python 3.11+
- PostgreSQL 14+
- `yt-dlp` (for video scraping)

### Installation

```bash
git clone https://github.com/shaincarrasco/climbing-ml.git
cd climbing-ml

pip install -r requirements.txt

# Create database and run migrations
createdb climbing_platform
psql climbing_platform < db/schema.sql
psql climbing_platform < db/seed.sql

# Set DB connection (optional — defaults to localhost/climbing_platform)
echo "DATABASE_URL=postgresql://localhost/climbing_platform" > .env
```

### Running the API

```bash
python3 api/app.py
# → http://localhost:5001
```

### Retrain Models

Run after scraping new videos or syncing route data:

```bash
python3 retrain.py              # full pipeline (~10–20 min)
python3 retrain.py --quick      # pose model only (~30s)
python3 retrain.py --pose-only  # skip main model
```

The retrain pipeline:
1. Purges low-quality pose frames (data quality score < 35)
2. Aggregates pose_frames → pose_features.csv
3. Retrains pose imputer (hold geometry → pose metrics for all 57K routes)
4. Retrains hold type classifier
5. Retrains main difficulty model with full-coverage pose fusion + quantile regressors

## API

### `POST /api/predict`

Predict difficulty for a custom route.

```json
{
  "holds": [
    {"x": 12, "y": 4, "role": "hand"},
    {"x": 10, "y": 8, "role": "foot"}
  ],
  "board_angle": 40
}
```

Response:

```json
{
  "grade": "V5",
  "difficulty_score": 18.3,
  "grade_range": {"lo": "V4", "hi": "V6"},
  "confidence": 0.78
}
```

### `GET /api/routes`

Returns paginated routes with filters: `grade`, `angle`, `setter`, `sort`.

### `GET /api/routes/<id>`

Route detail with hold positions, grade history, and pose metrics (if available).

### `GET /api/pose-correlations`

Returns Pearson correlations between pose metrics and grade, used for the correlation explorer.

## Data Sources

- **Kilter Board routes**: downloaded via [BoardLib](https://github.com/lemeryfertitta/BoardLib)
- **Hold type labels**: [Climbology project](https://github.com/Rundstedtzz/climbology) (MIT License) — manually annotated hold types for the Kilter Original layout

## Key Design Decisions

**Why two-stage imputation?** With only ~1.6% of routes having video pose data, XGBoost ignores pose features (98.4% NaN means no signal). Imputing from hold geometry gives every route a predicted pose profile — and every new scraped video improves the imputer's predictions for *all* 57K routes.

**Why drop noisy imputer predictions?** Angular joint metrics (elbow, knee, shoulder angles) have CV MAE of 2–19° after imputation. These add noise rather than signal to the main model. Only metrics with CV MAE < 0.5 (normalized velocity, tension, reach asymmetry) are passed to the difficulty model.

**Grade-balanced sample weights:** V3–V6 routes make up ~60% of the training set. Inverse-frequency weights (softened with 0.6 power) prevent the model from over-fitting to mid-grade routes at the cost of accuracy on V0–V2 and V9+.

**Prediction caching:** Route difficulty prediction is deterministic given hold positions + board angle. An in-memory MD5 cache (20K entries) eliminates recomputation for repeated predictions in the creator UI.
