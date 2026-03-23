"""
pipeline/pose_extractor.py
--------------------------
Batch video → MediaPipe Pose Landmarker (Tasks API) → pose_frames (PostgreSQL)

Per-frame metrics extracted
---------------------------
  Angles (degrees):
    hip_angle_deg          shoulder-hip-knee (body angle / compression)
    shoulder_rot_deg       tilt of shoulder line vs horizontal
    elbow_l_deg            left  shoulder→elbow→wrist
    elbow_r_deg            right shoulder→elbow→wrist
    knee_l_deg             left  hip→knee→ankle
    knee_r_deg             right hip→knee→ankle
    shoulder_l_deg         left  hip→shoulder→elbow (shoulder flexion)
    shoulder_r_deg         right hip→shoulder→elbow
    hip_spread_deg         angle between left/right leg vectors (drop-knee / flag)
    left_foot_angle_deg    left  knee→ankle→toe
    right_foot_angle_deg   right knee→ankle→toe

  Reach / spatial:
    left_arm_reach_norm    left  shoulder→wrist / torso height  (0–1.5)
    right_arm_reach_norm   right shoulder→wrist / torso height
    com_height_norm        1 - COM_y  (0=low, 1=high in frame)
    foot_hand_height_diff  avg_hand_y − avg_foot_y  (positive = feet higher on wall)
    com_to_hands_dist      normalized dist from COM to mid-wrist

  Body-tension:
    tension_score          composite body-tightness proxy (0–1)
    is_straight_arm_l      elbow_l_deg > 160°
    is_straight_arm_r      elbow_r_deg > 160°

  Velocity (inter-frame, NULL for first frame of each attempt):
    com_velocity           COM movement speed (norm-coords / sec)
    hand_l_velocity        left  wrist speed
    hand_r_velocity        right wrist speed
    elbow_l_ang_vel        left  elbow  deg/sec
    elbow_r_ang_vel        right elbow  deg/sec
    knee_l_ang_vel         left  knee   deg/sec
    knee_r_ang_vel         right knee   deg/sec
    hip_ang_vel            hip angle    deg/sec
    shoulder_l_ang_vel     left  shoulder deg/sec
    shoulder_r_ang_vel     right shoulder deg/sec

Requires:
    mediapipe>=0.10  numpy<2  opencv-python  psycopg2-binary
    ml/pose_landmarker_full.task  (auto-downloaded if missing)

Usage:
    python pipeline/pose_extractor.py --video path/to/clip.mp4 [--dry-run]
    python pipeline/pose_extractor.py --video-dir path/to/videos/
    python pipeline/pose_extractor.py --stats
    python pipeline/pose_extractor.py --summarize <attempt-uuid>
"""

import argparse
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import psycopg2
from dotenv import load_dotenv

from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmark as LM

# ── Model path ────────────────────────────────────────────────────────────────
_BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_PATH = _BASE_DIR / "ml" / "pose_landmarker_full.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)


def _ensure_model():
    if not MODEL_PATH.exists():
        import urllib.request
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pose model → {MODEL_PATH} …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


# ── 17 key joints ─────────────────────────────────────────────────────────────
CLIMBING_JOINT_NAMES = [
    "nose",
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_hip",       "right_hip",
    "left_knee",      "right_knee",
    "left_ankle",     "right_ankle",
    "left_index",     "right_index",
    "left_foot_index","right_foot_index",
]

CLIMBING_JOINT_IDX = {
    "nose":              LM.NOSE.value,
    "left_shoulder":     LM.LEFT_SHOULDER.value,
    "right_shoulder":    LM.RIGHT_SHOULDER.value,
    "left_elbow":        LM.LEFT_ELBOW.value,
    "right_elbow":       LM.RIGHT_ELBOW.value,
    "left_wrist":        LM.LEFT_WRIST.value,
    "right_wrist":       LM.RIGHT_WRIST.value,
    "left_hip":          LM.LEFT_HIP.value,
    "right_hip":         LM.RIGHT_HIP.value,
    "left_knee":         LM.LEFT_KNEE.value,
    "right_knee":        LM.RIGHT_KNEE.value,
    "left_ankle":        LM.LEFT_ANKLE.value,
    "right_ankle":       LM.RIGHT_ANKLE.value,
    "left_index":        LM.LEFT_INDEX.value,
    "right_index":       LM.RIGHT_INDEX.value,
    "left_foot_index":   LM.LEFT_FOOT_INDEX.value,
    "right_foot_index":  LM.RIGHT_FOOT_INDEX.value,
}

SKELETON_CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("left_wrist",     "left_index"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("right_wrist",    "right_index"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("left_ankle",     "left_foot_index"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("right_ankle",    "right_foot_index"),
    ("nose",           "left_shoulder"),
    ("nose",           "right_shoulder"),
]


# ── DB connection ─────────────────────────────────────────────────────────────
def get_conn():
    load_dotenv()
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME",     "climbing_platform"),
        user=os.getenv("DB_USER",       "shaincarrasco"),
        password=os.getenv("DB_PASSWORD", "") or None,
        host=os.getenv("DB_HOST",       "localhost"),
        port=os.getenv("DB_PORT",       "5432"),
    )


# ── Geometry helpers ──────────────────────────────────────────────────────────
def _lm_xy(lm_list, idx: int):
    lm = lm_list[idx]
    return lm.x, lm.y


def _angle_deg(a, b, c) -> float:
    """Angle at vertex b formed by rays b→a and b→c."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = math.hypot(*ba) * math.hypot(*bc)
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def _angle_between_vecs(v1, v2) -> float:
    """Angle between two 2-D vectors (degrees)."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag = math.hypot(*v1) * math.hypot(*v2)
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ── Per-frame metrics (no inter-frame data needed) ────────────────────────────
def extract_metrics(pose_landmarks: list) -> dict:
    """
    Returns all per-frame metrics from a single MediaPipe pose result.
    Velocity fields are NOT included here — they are added in process_video()
    using the previous frame.
    """
    def xy(idx): return _lm_xy(pose_landmarks, idx)

    l_sh  = xy(LM.LEFT_SHOULDER.value)
    r_sh  = xy(LM.RIGHT_SHOULDER.value)
    l_hip = xy(LM.LEFT_HIP.value)
    r_hip = xy(LM.RIGHT_HIP.value)
    l_kn  = xy(LM.LEFT_KNEE.value)
    r_kn  = xy(LM.RIGHT_KNEE.value)
    l_wr  = xy(LM.LEFT_WRIST.value)
    r_wr  = xy(LM.RIGHT_WRIST.value)
    l_ank = xy(LM.LEFT_ANKLE.value)
    r_ank = xy(LM.RIGHT_ANKLE.value)
    l_el  = xy(LM.LEFT_ELBOW.value)
    r_el  = xy(LM.RIGHT_ELBOW.value)
    l_ft  = xy(LM.LEFT_FOOT_INDEX.value)
    r_ft  = xy(LM.RIGHT_FOOT_INDEX.value)

    mid_sh  = ((l_sh[0]  + r_sh[0])  / 2, (l_sh[1]  + r_sh[1])  / 2)
    mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
    mid_kn  = ((l_kn[0]  + r_kn[0])  / 2, (l_kn[1]  + r_kn[1])  / 2)
    mid_wr  = ((l_wr[0]  + r_wr[0])  / 2, (l_wr[1]  + r_wr[1])  / 2)

    # ── Existing metrics ──────────────────────────────────────────────────────
    hip_angle    = _angle_deg(mid_sh, mid_hip, mid_kn)
    sh_dx        = r_sh[0] - l_sh[0]
    sh_dy        = r_sh[1] - l_sh[1]
    shoulder_rot = math.degrees(math.atan2(sh_dy, sh_dx))
    torso_h      = _dist(mid_sh, mid_hip) or 1.0
    l_arm_reach  = _dist(l_sh, l_wr) / torso_h
    r_arm_reach  = _dist(r_sh, r_wr) / torso_h
    l_foot_angle = _angle_deg(l_kn, l_ank, l_ft)
    r_foot_angle = _angle_deg(r_kn, r_ank, r_ft)
    com_y        = (mid_hip[1] + mid_sh[1]) / 2
    com_height_norm = round(1.0 - com_y, 4)
    avg_arm_ext  = (l_arm_reach + r_arm_reach) / 2
    hip_sag      = max(0.0, mid_hip[1] - mid_sh[1])
    tension_score = max(0.0, min(1.0, 1.0 - (avg_arm_ext * 0.4 + hip_sag * 2.0)))

    # ── New joint angles ──────────────────────────────────────────────────────
    # Elbow: shoulder→elbow→wrist
    elbow_l = _angle_deg(l_sh,  l_el, l_wr)
    elbow_r = _angle_deg(r_sh,  r_el, r_wr)

    # Knee: hip→knee→ankle
    knee_l  = _angle_deg(l_hip, l_kn, l_ank)
    knee_r  = _angle_deg(r_hip, r_kn, r_ank)

    # Shoulder flexion: hip→shoulder→elbow (arm vs torso)
    shoulder_l = _angle_deg(l_hip, l_sh, l_el)
    shoulder_r = _angle_deg(r_hip, r_sh, r_el)

    # Hip spread: angle between left and right leg vectors from each hip
    l_leg_vec = (l_kn[0] - l_hip[0], l_kn[1] - l_hip[1])
    r_leg_vec = (r_kn[0] - r_hip[0], r_kn[1] - r_hip[1])
    hip_spread = _angle_between_vecs(l_leg_vec, r_leg_vec)

    # Foot-hand height diff: in normalised coords y↑ = lower on wall
    # Positive value means feet are HIGHER on wall than hands (high-feet climbing)
    avg_hand_y  = (l_wr[1] + r_wr[1]) / 2
    avg_foot_y  = (l_ft[1]  + r_ft[1])  / 2
    foot_hand_height_diff = round(avg_hand_y - avg_foot_y, 4)

    # COM → hands distance (normalised by torso height for scale invariance)
    com_pos = ((mid_sh[0] + mid_hip[0]) / 2, (mid_sh[1] + mid_hip[1]) / 2)
    com_to_hands = round(_dist(com_pos, mid_wr) / torso_h, 4)

    # Straight-arm flags
    is_straight_l = elbow_l > 160.0
    is_straight_r = elbow_r > 160.0

    # ── Raw 17-joint snapshot ─────────────────────────────────────────────────
    raw = {}
    for name, idx in CLIMBING_JOINT_IDX.items():
        lm = pose_landmarks[idx]
        raw[name] = {
            "x": round(lm.x, 5),
            "y": round(lm.y, 5),
            "z": round(lm.z, 5),
            "visibility": round(getattr(lm, "visibility", 1.0), 3),
        }

    return {
        # ── angles ────────────────────────────────────────────────────────────
        "hip_angle_deg":          round(hip_angle,    2),
        "shoulder_rot_deg":       round(shoulder_rot, 2),
        "elbow_l_deg":            round(elbow_l,      2),
        "elbow_r_deg":            round(elbow_r,      2),
        "knee_l_deg":             round(knee_l,       2),
        "knee_r_deg":             round(knee_r,       2),
        "shoulder_l_deg":         round(shoulder_l,   2),
        "shoulder_r_deg":         round(shoulder_r,   2),
        "hip_spread_deg":         round(hip_spread,   2),
        "left_foot_angle_deg":    round(l_foot_angle, 2),
        "right_foot_angle_deg":   round(r_foot_angle, 2),
        # ── reach / spatial ───────────────────────────────────────────────────
        "left_arm_reach_norm":    round(min(l_arm_reach, 1.5), 4),
        "right_arm_reach_norm":   round(min(r_arm_reach, 1.5), 4),
        "com_height_norm":        com_height_norm,
        "foot_hand_height_diff":  foot_hand_height_diff,
        "com_to_hands_dist":      com_to_hands,
        # ── tension / body control ────────────────────────────────────────────
        "tension_score":          round(tension_score, 4),
        "is_straight_arm_l":      is_straight_l,
        "is_straight_arm_r":      is_straight_r,
        # ── raw landmarks (for future use) ────────────────────────────────────
        "raw_landmarks":          raw,
        # ── position snapshots used by velocity computation ───────────────────
        "_com_pos":    com_pos,
        "_l_wr_pos":   l_wr,
        "_r_wr_pos":   r_wr,
    }


# ── Inter-frame velocity computation ─────────────────────────────────────────
def compute_velocities(curr: dict, prev: dict, dt: float) -> dict:
    """
    Given two consecutive frame metric dicts and the time delta (seconds),
    return velocity metrics. All angular velocities in deg/sec;
    linear velocities in normalised-coord/sec.
    """
    if dt <= 0:
        return {}

    def _ang_vel(key):
        return round(abs(curr[key] - prev[key]) / dt, 2)

    def _lin_vel(curr_pos, prev_pos):
        return round(_dist(curr_pos, prev_pos) / dt, 4)

    return {
        "elbow_l_ang_vel":    _ang_vel("elbow_l_deg"),
        "elbow_r_ang_vel":    _ang_vel("elbow_r_deg"),
        "knee_l_ang_vel":     _ang_vel("knee_l_deg"),
        "knee_r_ang_vel":     _ang_vel("knee_r_deg"),
        "hip_ang_vel":        _ang_vel("hip_angle_deg"),
        "shoulder_l_ang_vel": _ang_vel("shoulder_l_deg"),
        "shoulder_r_ang_vel": _ang_vel("shoulder_r_deg"),
        "com_velocity":       _lin_vel(curr["_com_pos"],  prev["_com_pos"]),
        "hand_l_velocity":    _lin_vel(curr["_l_wr_pos"], prev["_l_wr_pos"]),
        "hand_r_velocity":    _lin_vel(curr["_r_wr_pos"], prev["_r_wr_pos"]),
    }


# ── Active climbing segment detection ────────────────────────────────────────
def filter_active_climbing_segment(
    rows: list,
    buffer_frames: int = 8,
    min_climb_frames: int = 8,
) -> list:
    """
    From a sequence of (attempt_id, ts, metrics) rows, detect and return only
    frames where active climbing is happening.

    Keeps ALL significant climbing segments (not just the last one), so
    multi-climb compilation videos and training videos work correctly.
    Non-climbing gaps (walking around, chalking, long rests) are dropped.

    Falls back to all frames if no clear segments are found or if the
    filter would remove more than 80% of frames (safety valve).
    """
    if len(rows) < min_climb_frames:
        return rows

    n = len(rows)
    com_heights = [m.get("com_height_norm", 0.5) for _, _, m in rows]

    # 7-frame smoothed COM height
    w = 7
    smoothed = []
    for i in range(n):
        s = max(0, i - w // 2)
        e = min(n, i + w // 2 + 1)
        smoothed.append(sum(com_heights[s:e]) / (e - s))

    look_ahead = min(20, max(10, n // 20))

    # A frame is "climbing" if:
    #   - COM rises over the next window (active upward movement), OR
    #   - tension is high and COM isn't falling fast (static grip on wall)
    is_climbing = []
    for i in range(n):
        future  = min(i + look_ahead, n - 1)
        past    = max(0, i - look_ahead)
        delta_f = smoothed[future] - smoothed[i]          # forward change
        delta_b = smoothed[i]     - smoothed[past]        # backward change
        tension = rows[i][2].get("tension_score", 0.0) or 0.0
        active  = (delta_f > 0.01                          # COM rising ahead
                   or delta_b > 0.01                       # COM rose recently
                   or (tension > 0.35 and delta_f > -0.02) # high tension, stable
                   )
        is_climbing.append(active)

    # Collect ALL contiguous climbing segments
    segments = []
    in_seg, seg_start = False, 0
    for i, c in enumerate(is_climbing):
        if c and not in_seg:
            in_seg, seg_start = True, i
        elif not c and in_seg:
            in_seg = False
            if i - seg_start >= min_climb_frames:
                segments.append((seg_start, i))
    if in_seg and n - seg_start >= min_climb_frames:
        segments.append((seg_start, n))

    if not segments:
        return rows  # no clear segments — keep everything

    # Build index set of all frames inside any segment (+ buffer)
    keep = set()
    for s, e in segments:
        for idx in range(max(0, s - buffer_frames), min(n, e + buffer_frames)):
            keep.add(idx)

    kept = [rows[i] for i in sorted(keep)]

    # Safety valve: if filter removes >80% of frames, something went wrong
    if len(kept) < n * 0.20:
        print(f"    [climb-filter] filter too aggressive ({len(kept)}/{n}) — keeping all")
        return rows

    seg_summary = f"{len(segments)} segment(s)"
    print(f"    [climb-filter] {len(kept)}/{n} frames kept ({seg_summary})")
    return kept


# ── Core video processor ──────────────────────────────────────────────────────
def process_video(
    video_path: str,
    attempt_id: str,
    sample_fps: float = 10.0,
    min_confidence: float = 0.5,
    dry_run: bool = False,
    verbose: bool = True,
    climb_uuid: str | None = None,
    filter_climbing: bool = False,
) -> int:
    """Extract pose frames from a video. Returns number of rows inserted."""
    _ensure_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frm  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration   = total_frm / src_fps
    frame_step = max(1, int(round(src_fps / sample_fps)))

    if verbose:
        print(f"  Video  : {Path(video_path).name}")
        print(f"  FPS    : {src_fps:.1f}  Duration: {duration:.1f}s  Frames: {total_frm}")
        print(f"  Sample : every {frame_step} frames (~{src_fps/frame_step:.1f} fps)")

    base_opts = _mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    options   = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    rows       = []   # list of (attempt_id, ts, metrics_with_velocities)
    prev_m     = None
    prev_ts    = None

    with mp_vision.PoseLandmarker.create_from_options(options) as detector:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_step == 0:
                ts_ms = int(frame_idx / src_fps * 1000)
                ts    = round(frame_idx / src_fps, 4)

                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mpi   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect_for_video(mpi, ts_ms)

                if result.pose_landmarks:
                    m = extract_metrics(result.pose_landmarks[0])

                    # Add velocity metrics from previous frame
                    if prev_m is not None and prev_ts is not None:
                        dt = ts - prev_ts
                        m.update(compute_velocities(m, prev_m, dt))

                    rows.append((attempt_id, ts, m))
                    prev_m  = m
                    prev_ts = ts

            frame_idx += 1

    cap.release()

    if verbose:
        sampled = total_frm // frame_step
        print(f"  Detected: {len(rows)} / {sampled} sampled frames "
              f"({100*len(rows)/max(sampled,1):.0f}%)")

    # Filter to active climbing segment only (strips non-climbing footage)
    if filter_climbing and rows:
        rows = filter_active_climbing_segment(rows)

    if dry_run:
        if verbose:
            print("  [dry-run] Not writing to DB.")
            if rows:
                sample = {k: v for k, v in rows[0][2].items()
                          if not k.startswith("_") and k != "raw_landmarks"}
                print("  Sample metrics:", json.dumps(sample, indent=2))
        return len(rows)

    if not rows:
        return 0

    conn = get_conn()
    cur  = conn.cursor()

    for (att_id, ts, m) in rows:
        cur.execute("""
            INSERT INTO pose_frames (
                id, attempt_id, climb_uuid, timestamp_sec,
                hip_angle_deg, shoulder_rot_deg,
                elbow_l_deg, elbow_r_deg,
                knee_l_deg, knee_r_deg,
                shoulder_l_deg, shoulder_r_deg,
                hip_spread_deg,
                left_arm_reach_norm, right_arm_reach_norm,
                left_foot_angle_deg, right_foot_angle_deg,
                com_height_norm, foot_hand_height_diff,
                com_to_hands_dist, tension_score,
                is_straight_arm_l, is_straight_arm_r,
                com_velocity, hand_l_velocity, hand_r_velocity,
                elbow_l_ang_vel, elbow_r_ang_vel,
                knee_l_ang_vel, knee_r_ang_vel,
                hip_ang_vel, shoulder_l_ang_vel, shoulder_r_ang_vel,
                raw_landmarks
            ) VALUES (
                %s,%s,%s,%s,
                %s,%s, %s,%s, %s,%s, %s,%s, %s,
                %s,%s, %s,%s, %s,%s, %s,%s, %s,%s,
                %s,%s,%s, %s,%s, %s,%s, %s,%s,%s,
                %s
            )
            ON CONFLICT DO NOTHING
        """, (
            str(uuid.uuid4()), att_id, climb_uuid, ts,
            m.get("hip_angle_deg"),       m.get("shoulder_rot_deg"),
            m.get("elbow_l_deg"),         m.get("elbow_r_deg"),
            m.get("knee_l_deg"),          m.get("knee_r_deg"),
            m.get("shoulder_l_deg"),      m.get("shoulder_r_deg"),
            m.get("hip_spread_deg"),
            m.get("left_arm_reach_norm"), m.get("right_arm_reach_norm"),
            m.get("left_foot_angle_deg"), m.get("right_foot_angle_deg"),
            m.get("com_height_norm"),     m.get("foot_hand_height_diff"),
            m.get("com_to_hands_dist"),   m.get("tension_score"),
            m.get("is_straight_arm_l"),   m.get("is_straight_arm_r"),
            m.get("com_velocity"),        m.get("hand_l_velocity"),
            m.get("hand_r_velocity"),
            m.get("elbow_l_ang_vel"),     m.get("elbow_r_ang_vel"),
            m.get("knee_l_ang_vel"),      m.get("knee_r_ang_vel"),
            m.get("hip_ang_vel"),
            m.get("shoulder_l_ang_vel"),  m.get("shoulder_r_ang_vel"),
            json.dumps(m["raw_landmarks"]),
        ))

    conn.commit()
    cur.close()
    conn.close()

    if verbose:
        print(f"  Inserted {len(rows):,} pose_frames rows.")

    return len(rows)


# ── Batch directory ───────────────────────────────────────────────────────────
def process_directory(
    video_dir: str,
    attempt_id_map: Optional[dict] = None,
    sample_fps: float = 10.0,
    min_confidence: float = 0.5,
    dry_run: bool = False,
    filter_climbing: bool = False,
) -> dict:
    video_dir  = Path(video_dir)
    extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    videos     = sorted(f for f in video_dir.iterdir() if f.suffix.lower() in extensions)

    print(f"\nFound {len(videos)} video(s) in {video_dir}")
    if filter_climbing:
        print("  Active climbing filter: ON (will strip non-climbing footage)\n")
    else:
        print()

    results = {}
    for vf in videos:
        name   = vf.name
        att_id = (attempt_id_map or {}).get(name) or str(uuid.uuid4())
        print(f"── {name}  →  attempt {att_id}")
        try:
            n = process_video(
                str(vf), att_id,
                sample_fps=sample_fps,
                min_confidence=min_confidence,
                dry_run=dry_run,
                filter_climbing=filter_climbing,
            )
            results[name] = {"attempt_id": att_id, "frames": n, "status": "ok"}
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"attempt_id": att_id, "frames": 0, "status": str(e)}
        print()

    return results


# ── DB stats ──────────────────────────────────────────────────────────────────
def print_stats():
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM pose_frames")
    total = cur.fetchone()[0]
    cur.execute("""
        SELECT attempt_id, COUNT(*) AS frames,
               MIN(timestamp_sec), MAX(timestamp_sec)
        FROM pose_frames
        GROUP BY attempt_id
        ORDER BY frames DESC
        LIMIT 20
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    print(f"\npose_frames total: {total:,}\n")
    if rows:
        print(f"{'attempt_id':<38} {'frames':>7} {'duration':>10}")
        print("─" * 60)
        for att, frm, t0, t1 in rows:
            dur = f"{t1-t0:.1f}s" if t0 is not None and t1 is not None else "—"
            print(f"{str(att):<38} {frm:>7,} {dur:>10}")
    else:
        print("No pose data yet.")


# ── Aggregate metrics for one attempt ────────────────────────────────────────
def summarize_attempt(attempt_id: str) -> dict:
    """
    Pull all pose_frames for an attempt and return per-metric stats.
    Covers all new metrics including velocities and joint angles.
    """
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute("""
        SELECT
            hip_angle_deg, shoulder_rot_deg,
            elbow_l_deg, elbow_r_deg,
            knee_l_deg, knee_r_deg,
            shoulder_l_deg, shoulder_r_deg,
            hip_spread_deg,
            left_arm_reach_norm, right_arm_reach_norm,
            left_foot_angle_deg, right_foot_angle_deg,
            com_height_norm, foot_hand_height_diff,
            com_to_hands_dist, tension_score,
            is_straight_arm_l, is_straight_arm_r,
            com_velocity, hand_l_velocity, hand_r_velocity,
            elbow_l_ang_vel, elbow_r_ang_vel,
            knee_l_ang_vel, knee_r_ang_vel,
            hip_ang_vel, shoulder_l_ang_vel, shoulder_r_ang_vel
        FROM pose_frames
        WHERE attempt_id = %s
        ORDER BY timestamp_sec
    """, (attempt_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return {}

    cols = [
        "hip_angle_deg", "shoulder_rot_deg",
        "elbow_l_deg", "elbow_r_deg",
        "knee_l_deg", "knee_r_deg",
        "shoulder_l_deg", "shoulder_r_deg",
        "hip_spread_deg",
        "left_arm_reach_norm", "right_arm_reach_norm",
        "left_foot_angle_deg", "right_foot_angle_deg",
        "com_height_norm", "foot_hand_height_diff",
        "com_to_hands_dist", "tension_score",
        "is_straight_arm_l", "is_straight_arm_r",
        "com_velocity", "hand_l_velocity", "hand_r_velocity",
        "elbow_l_ang_vel", "elbow_r_ang_vel",
        "knee_l_ang_vel", "knee_r_ang_vel",
        "hip_ang_vel", "shoulder_l_ang_vel", "shoulder_r_ang_vel",
    ]

    bool_cols = {"is_straight_arm_l", "is_straight_arm_r"}

    def _stats(vals):
        n = len(vals)
        if not n:
            return {}
        return {
            "mean": round(sum(vals) / n, 3),
            "min":  round(vals[0], 3),
            "p50":  round(vals[n // 2], 3),
            "p90":  round(vals[int(n * 0.9)], 3),
            "max":  round(vals[-1], 3),
        }

    result = {}
    for i, col in enumerate(cols):
        raw_vals = [r[i] for r in rows if r[i] is not None]
        if not raw_vals:
            continue
        if col in bool_cols:
            result[col] = {"pct_true": round(sum(raw_vals) / len(raw_vals), 3)}
        else:
            result[col] = _stats(sorted(raw_vals))

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Climbing pose extractor (MediaPipe Tasks)")
    parser.add_argument("--video",          help="Path to a single video file")
    parser.add_argument("--video-dir",      help="Directory of video files to batch process")
    parser.add_argument("--attempt-id",     help="UUID to link pose_frames to (single video mode)")
    parser.add_argument("--attempt-id-map", help='JSON file: {"filename.mp4": "uuid", ...}')
    parser.add_argument("--sample-fps",     type=float, default=10.0)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--dry-run",         action="store_true")
    parser.add_argument("--filter-climbing", action="store_true",
                        help="Strip non-climbing footage (final send detection)")
    parser.add_argument("--stats",           action="store_true")
    parser.add_argument("--summarize",       metavar="ATTEMPT_UUID")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    if args.summarize:
        summary = summarize_attempt(args.summarize)
        print(json.dumps(summary, indent=2))
        return

    if args.video:
        att_id = args.attempt_id or str(uuid.uuid4())
        if not args.attempt_id:
            print(f"[info] No --attempt-id supplied → using {att_id}")
        process_video(
            args.video, att_id,
            sample_fps=args.sample_fps,
            min_confidence=args.min_confidence,
            dry_run=args.dry_run,
            verbose=True,
            filter_climbing=args.filter_climbing,
        )
        return

    if args.video_dir:
        id_map = None
        if args.attempt_id_map:
            with open(args.attempt_id_map) as f:
                id_map = json.load(f)
        results = process_directory(
            args.video_dir, id_map,
            sample_fps=args.sample_fps,
            min_confidence=args.min_confidence,
            dry_run=args.dry_run,
            filter_climbing=args.filter_climbing,
        )
        ok  = sum(1 for r in results.values() if r["status"] == "ok")
        err = len(results) - ok
        tot = sum(r["frames"] for r in results.values())
        print(f"\n── Summary ──────────────────────────────────")
        print(f"  {ok} ok  |  {err} errors  |  {tot:,} total frames stored")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
