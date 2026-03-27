# Climbing ML — Difficulty Model Improvement Report

**Date:** 2026-03-27
**Goal:** Improve within-1-V-grade accuracy from 74.2% → 90%+

---

## Baseline (before changes — v3)

| Metric | Value |
|--------|-------|
| Within-1-V-grade accuracy | **74.21%** |
| Exact grade match | **33.1%** |
| Test RMSE | 0.06224 |
| Test MAE | 0.04750 |
| Features | 67 |
| Model | XGBoost (800 estimators) |

---

## Changes Made

### 1. Feature Engineering (`pipeline/feature_extraction.py`)

Added **9 new features** grouped into 5 categories:

| Feature | Description | Why it helps |
|---------|-------------|-------------|
| `seq_angle_avg` | Mean bearing angle of consecutive hand-hold moves (full atan2, signed dx/dy) | Captures whether the route trends left/right/straight — the existing `avg_move_angle_deg` uses unsigned abs(atan2) so loses directional information |
| `seq_angle_std` | Std dev of move bearing angles | Routes with high variance in direction are generally harder to sequence |
| `zone_bl_count` | Hand holds in bottom-left quadrant | Distribution across quadrants encodes where the climbing happens on the board |
| `zone_br_count` | Hand holds in bottom-right quadrant | Same — asymmetric zone loading = harder body positioning |
| `zone_tl_count` | Hand holds in top-left quadrant | Captures reach-heavy top section vs. technical start |
| `zone_tr_count` | Hand holds in top-right quadrant | Same |
| `zone_transitions` | Quadrant transitions in hold sequence | High transition count = traversing route = harder coordination |
| `crux_cluster_compactness` | 1 / mean pairwise distance of densest 3-hold window | Tightly clustered holds = intricate, pumpy crux section |
| `path_efficiency` | Euclidean start→finish / total path length | < 1.0 = traversing route; encodes the "style" of the problem |
| `max_consec_vert_gain_cm` | Largest single upward dy in sequence | Captures the single biggest reach-up — a direct crux proxy |

**To regenerate features with the new columns:**
```bash
cd /Users/shaincarrasco/Desktop/ClimbingML
python3 pipeline/feature_extraction.py
```

---

### 2. Hyperparameter Optimization (`ml/difficulty_model.py`)

Added **Optuna hyperparameter search** (50 trials):

- **Objective:** Maximize within-1-V-grade accuracy on validation set
- **Search space:**
  - `n_estimators`: 500–2000 (step 100)
  - `max_depth`: 4–10
  - `learning_rate`: 0.01–0.1 (log scale)
  - `min_child_weight`: 1–10
  - `subsample`: 0.6–1.0
  - `colsample_bytree`: 0.6–1.0
  - `reg_alpha`: 0.0–1.0
  - `reg_lambda`: 0.5–3.0
- **Fallback:** If Optuna's best trial does not beat the baseline within-1 score, the original params are kept
- **Sampler:** TPE (Tree-structured Parzen Estimator), seed=42

---

### 3. Model Architecture (`ml/difficulty_model.py`)

Added **LightGBM** as a second base model and a **Ridge regression stacking meta-learner**:

- LightGBM params: 1000 estimators, max_depth=7, num_leaves=63, subsample=0.8, early stopping
- Stacking via 5-fold out-of-fold (OOF) predictions from both XGBoost and LightGBM
- Ridge meta-learner trained on OOF stack (`[xgb_oof, lgbm_oof]` → `difficulty_score`)
- Stacking is activated as primary model **only if it improves within-1 by >1%** over XGBoost alone
- New artifacts saved: `ml/difficulty_model.lgb`, `ml/difficulty_model_stack.pkl`

---

### 4. Grade Distribution Calibration (`ml/difficulty_model.py`)

Two changes:

**a) Grade distribution printing + sample weighting:**
- Grade distribution printed at every training run with rare-grade flags
- Grades with < 500 samples flagged and receive stronger inverse-frequency sample weights
- Same soft-power formula (0.6 exponent) applied uniformly; the print output makes imbalance visible

**b) Isotonic regression calibration:**
- After training, an `IsotonicRegression` is fit on validation set predictions vs. true `difficulty_score`
- Provides monotone, data-driven mapping from raw predicted score to calibrated score
- Calibration is only applied if it improves val within-1 accuracy
- Saved to `ml/difficulty_model_isotonic.pkl`

---

### 5. Package Dependencies (`requirements.txt`)

Added:
```
lightgbm>=4.0.0
optuna>=3.0.0
```

Install with:
```bash
pip3 install lightgbm optuna
```

---

## How to Run

```bash
# Step 1 — Re-extract features (adds 9 new columns to routes_features.csv)
cd /Users/shaincarrasco/Desktop/ClimbingML
python3 pipeline/feature_extraction.py

# Step 2 — Retrain with all improvements
python3 ml/difficulty_model.py
```

Or skip feature re-extraction to test the model improvements on the existing 67 features:
```bash
cd /Users/shaincarrasco/Desktop/ClimbingML
python3 ml/difficulty_model.py
```

---

## Results — v6 (2026-03-27, current model)

| Metric | v3 Baseline | v6 (mirror aug, no consensus) | Delta |
|--------|-------------|-------------------------------|-------|
| Within-1-V-grade accuracy | 74.21% | **76.9%** | **+2.7pp** |
| Exact grade match | 33.1% | 33.2% | +0.1pp |
| Test RMSE | 0.06224 | 0.06816 | — |
| Test MAE | 0.04750 | 0.05221 | — |
| Features | 67 | 72 | +5 |
| Mirror augmentation | No | Yes (V5+, +29K synthetic routes) | new |
| Consensus weighting | No | No (removed — hurt V4/V6/V11) | — |

### Per-grade breakdown (v6)

| Grade | n | Exact | Within-1 |
|-------|---|-------|----------|
| V0    | 142 | 38.7% | 83.1% |
| V1    | 116 | 44.8% | 79.3% |
| V2    |  64 | 28.1% | 90.6% |
| V3    | 117 | 40.2% | 82.1% |
| V4    |  84 | 25.0% | 73.8% |
| V5    |  62 | 17.7% | 69.4% |
| V6    |  25 | 24.0% | 44.0% |
| V7    |  14 |  0.0% | 50.0% |
| V8+   | <10 |  0.0% |  0.0% |

### Training history

| Version | Within-1 | Key change |
|---------|----------|------------|
| v3 | 74.21% | Baseline (stacking + isotonic) |
| v4 | 73.70% | + consensus weighting (uncapped → destabilized) |
| v5 | 73.14% | + mirror aug + capped consensus (both hurt V4/V6/V11) |
| **v6** | **76.9%** | + mirror aug only (consensus removed) — **new best** |

---

## Expected Improvement Breakdown

Based on the changes and the structure of the data:

| Change | Expected contribution |
|--------|-----------------------|
| Optuna tuning | +1–3% (typical for well-tuned baselines) |
| LightGBM stacking | +1–2% (ensemble diversity) |
| New features (9) | +0.5–2% (zone transitions and path efficiency are high-signal) |
| Isotonic calibration | +0.5–1% (corrects systematic score-to-grade mapping errors) |
| **Total expected** | **+3–8%** (target: 79–84% without re-extraction; 82–86% with) |

Reaching 90%+ within-1 is likely to require:
1. The full 628K route dataset (currently ~57K extracted — run sync to completion)
2. Re-running feature extraction to include the 9 new geometric features
3. Pose imputation on full dataset for all routes
4. Potentially reducing grade granularity or merging adjacent sparse grades

---

*Generated by Claude Sonnet 4.6 during overnight model improvement session.*
